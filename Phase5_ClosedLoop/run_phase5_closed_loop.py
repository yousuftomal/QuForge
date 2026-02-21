#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class EncoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class TwinEncoder(nn.Module):
    def __init__(self, geom_in: int, phys_in: int, hidden_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.geom_encoder = EncoderMLP(geom_in, hidden_dim, emb_dim, dropout)
        self.phys_encoder = EncoderMLP(phys_in, hidden_dim, emb_dim, dropout)

    def encode_geom(self, x: torch.Tensor) -> torch.Tensor:
        return self.geom_encoder(x)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n <= 1e-12, 1.0, n)
    return x / n


def invert_target_transform(y_model: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    y = y_model.copy()
    if bool(transform["use_log"]):
        y = np.power(10.0, y)
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    return np.clip(y, lo, hi)


def predict_target_quantiles_bulk(models: Dict[str, Any], target: str, x: np.ndarray, transform: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10_m = models[f"{target}_q10"].predict(x)
    q50_m = models[f"{target}_q50"].predict(x)
    q90_m = models[f"{target}_q90"].predict(x)

    q10 = invert_target_transform(q10_m, transform)
    q50 = invert_target_transform(q50_m, transform)
    q90 = invert_target_transform(q90_m, transform)

    stacked = np.vstack([q10, q50, q90]).T
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def _apply_source_profile_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    source_calibration: Mapping[str, Any],
    target: str,
    source_profile: str,
    uncertainty_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    out10 = q10.copy()
    out50 = q50.copy()
    out90 = q90.copy()

    targets = source_calibration.get("targets", {}) if isinstance(source_calibration, dict) else {}
    tcal = targets.get(target, {}) if isinstance(targets, dict) else {}
    offsets = tcal.get("offsets_log10", {}) if isinstance(tcal, dict) else {}

    offset = 0.0
    if source_profile and source_profile != "global" and isinstance(offsets, dict):
        offset = float(offsets.get(source_profile, 0.0))
    if abs(offset) > 0:
        scale = 10.0 ** offset
        out10 = out10 * scale
        out50 = out50 * scale
        out90 = out90 * scale

    if source_profile == "global" and isinstance(tcal, dict):
        spread = float(tcal.get("offset_std_log10", 0.0))
        spread = max(0.0, spread * max(0.0, float(uncertainty_scale)))
        if spread > 0:
            factor = 10.0 ** spread
            out10 = np.minimum(out10, out50 / factor)
            out90 = np.maximum(out90, out50 * factor)

    stacked = np.vstack([out10, out50, out90]).T
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 5 closed-loop candidate engine")

    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--phase2-bundle", type=Path, default=root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument("--phase1-model", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-meta", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")
    parser.add_argument("--phase4-bundle", type=Path, default=root / "Phase4_Coherence" / "artifacts" / "phase4_coherence_bundle.joblib")

    parser.add_argument("--targets-csv", type=Path, default=None)
    parser.add_argument("--num-targets", type=int, default=24)
    parser.add_argument("--freq-min", type=float, default=4.0)
    parser.add_argument("--freq-max", type=float, default=6.0)
    parser.add_argument("--anh-min", type=float, default=-0.30)
    parser.add_argument("--anh-max", type=float, default=-0.14)

    parser.add_argument("--library-split", choices=("train", "all"), default="train")
    parser.add_argument("--retrieval-top-k", type=int, default=80)
    parser.add_argument("--top-n-per-target", type=int, default=8)

    parser.add_argument("--population-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--elite-fraction", type=float, default=0.22)
    parser.add_argument("--explore-fraction", type=float, default=0.12)
    parser.add_argument("--start-mutation-scale", type=float, default=0.15)
    parser.add_argument("--end-mutation-scale", type=float, default=0.02)

    parser.add_argument("--surrogate-weight", type=float, default=1.0)
    parser.add_argument("--embedding-weight", type=float, default=0.05)
    parser.add_argument("--robustness-weight", type=float, default=0.20)
    parser.add_argument("--robust-samples", type=int, default=5)
    parser.add_argument("--fabrication-tolerance", type=float, default=0.05)
    parser.add_argument("--robust-std-weight", type=float, default=0.30)

    parser.add_argument("--min-t1-p10-us", type=float, default=0.18)
    parser.add_argument("--max-t1-width-multiplier", type=float, default=1.5)
    parser.add_argument("--max-t2-width-multiplier", type=float, default=1.5)
    parser.add_argument("--allow-ood", action="store_true")
    parser.add_argument("--allow-low-confidence", action="store_true")
    parser.add_argument("--allow-uncertain", action="store_true")
    parser.add_argument("--source-profile", type=str, default="global", help="Source profile for coherence projection (default: global canonical)")
    parser.add_argument("--source-uncertainty-scale", type=float, default=1.0, help="Multiplier for source-heterogeneity uncertainty inflation under global profile")

    parser.add_argument("--score-w-t1-p10", type=float, default=1.0)
    parser.add_argument("--score-w-t1-width", type=float, default=0.20)
    parser.add_argument("--score-w-objective", type=float, default=0.30)
    parser.add_argument("--score-w-robust", type=float, default=0.20)
    parser.add_argument("--score-w-ood", type=float, default=0.50)
    parser.add_argument("--score-w-low-confidence", type=float, default=0.30)

    parser.add_argument("--selected-top-n", type=int, default=30)
    parser.add_argument("--fallback-top-n", type=int, default=10)

    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def generate_default_targets(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state)
    n = max(1, int(args.num_targets))

    freq = rng.uniform(args.freq_min, args.freq_max, size=n)
    anh = rng.uniform(args.anh_min, args.anh_max, size=n)

    df = pd.DataFrame(
        {
            "target_id": [f"target_{i+1:03d}" for i in range(n)],
            "freq_01_GHz": freq,
            "anharmonicity_GHz": anh,
        }
    )
    return df


def load_targets(args: argparse.Namespace) -> pd.DataFrame:
    if args.targets_csv is None:
        return generate_default_targets(args)

    df = pd.read_csv(args.targets_csv)
    required = ["freq_01_GHz", "anharmonicity_GHz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"targets-csv missing columns: {missing}")

    if "target_id" not in df.columns:
        df = df.copy()
        df["target_id"] = [f"target_{i+1:03d}" for i in range(len(df))]

    return df


def maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def build_feature_rows(
    candidates: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_medians: Mapping[str, float],
) -> Tuple[np.ndarray, pd.DataFrame]:
    rows: List[List[float]] = []
    enriched = candidates.copy()

    if "EJ_EC_ratio" not in enriched.columns:
        enriched["EJ_EC_ratio"] = np.nan

    ej = pd.to_numeric(enriched.get("pred_EJ_GHz"), errors="coerce")
    ec = pd.to_numeric(enriched.get("pred_EC_GHz"), errors="coerce")
    ratio = np.where(np.abs(ec.to_numpy(dtype=float)) > 1e-12, ej.to_numpy(dtype=float) / ec.to_numpy(dtype=float), np.nan)
    ratio_existing = pd.to_numeric(enriched.get("EJ_EC_ratio"), errors="coerce").to_numpy(dtype=float)
    use_ratio = np.where(np.isfinite(ratio_existing), ratio_existing, ratio)
    enriched["EJ_EC_ratio"] = use_ratio

    for _, r in enriched.iterrows():
        f = {
            "pad_width_um": maybe_float(r.get("pad_width_um")),
            "pad_height_um": maybe_float(r.get("pad_height_um")),
            "gap_um": maybe_float(r.get("gap_um")),
            "junction_area_um2": maybe_float(r.get("junction_area_um2")),
            "freq_01_GHz": maybe_float(r.get("pred_freq_01_GHz")),
            "anharmonicity_GHz": maybe_float(r.get("pred_anharmonicity_GHz")),
            "EJ_GHz": maybe_float(r.get("pred_EJ_GHz")),
            "EC_GHz": maybe_float(r.get("pred_EC_GHz")),
            "charge_sensitivity_GHz": maybe_float(r.get("pred_charge_sensitivity_GHz")),
            "EJ_EC_ratio": maybe_float(r.get("EJ_EC_ratio")),
        }

        for c in feature_cols:
            if f.get(c) is None or not np.isfinite(float(f.get(c))):
                f[c] = float(feature_medians[c])

        rows.append([float(f[c]) for c in feature_cols])

    return np.asarray(rows, dtype=float), enriched


def run_phase3_targets(args: argparse.Namespace, targets: pd.DataFrame) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "Phase3_InverseDesign"))
    from engine import build_target_map, infer_active_target_cols, load_context, run_inverse_design

    ctx = load_context(
        root=root,
        single_csv=args.single_csv,
        phase2_bundle=args.phase2_bundle,
        phase1_model_path=args.phase1_model,
        phase1_meta_path=args.phase1_meta,
    )

    out_rows: List[Dict[str, Any]] = []

    for i, row in targets.reset_index(drop=True).iterrows():
        target_id = str(row["target_id"])
        f01 = float(row["freq_01_GHz"])
        anh = float(row["anharmonicity_GHz"])

        ej = maybe_float(row.get("EJ_GHz"))
        ec = maybe_float(row.get("EC_GHz"))
        charge = maybe_float(row.get("charge_sensitivity_GHz"))

        target_map = build_target_map(
            ctx,
            freq_01_GHz=f01,
            anharmonicity_GHz=anh,
            EJ_GHz=ej,
            EC_GHz=ec,
            charge_sensitivity_GHz=charge,
        )
        active_cols = infer_active_target_cols(
            ej_given=ej is not None,
            ec_given=ec is not None,
            charge_given=charge is not None,
        )

        payload = run_inverse_design(
            ctx,
            target_map=target_map,
            active_target_cols=active_cols,
            library_split=args.library_split,
            retrieval_top_k=args.retrieval_top_k,
            top_n=args.top_n_per_target,
            population_size=args.population_size,
            iterations=args.iterations,
            elite_fraction=args.elite_fraction,
            explore_fraction=args.explore_fraction,
            start_mutation_scale=args.start_mutation_scale,
            end_mutation_scale=args.end_mutation_scale,
            surrogate_weight=args.surrogate_weight,
            embedding_weight=args.embedding_weight,
            robustness_weight=args.robustness_weight,
            robust_samples=args.robust_samples,
            fabrication_tolerance=args.fabrication_tolerance,
            robust_std_weight=args.robust_std_weight,
            bounds_low_q=0.001,
            bounds_high_q=0.999,
            random_state=args.random_state + 97 * i,
        )

        for cand in payload.get("results", []):
            rec = {k: v for k, v in cand.items()}
            rec["target_id"] = target_id
            rec["target_freq_01_GHz"] = f01
            rec["target_anharmonicity_GHz"] = anh
            out_rows.append(rec)

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def phase4_score_candidates(args: argparse.Namespace, candidates: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(args.phase4_bundle)
    feature_cols = list(bundle["feature_cols"])
    feature_medians = {k: float(v) for k, v in dict(bundle["feature_medians"]).items()}
    target_transforms = dict(bundle["target_transforms"])

    x_raw, candidates = build_feature_rows(candidates, feature_cols, feature_medians)
    x_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
    x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

    models = bundle["models"]
    t1_p10, t1_p50, t1_p90 = predict_target_quantiles_bulk(models, "t1_us", x_scaled, target_transforms["t1_us"])
    t2_p10, t2_p50, t2_p90 = predict_target_quantiles_bulk(models, "t2_us", x_scaled, target_transforms["t2_us"])

    source_calibration = bundle.get("source_calibration", {})
    t1_p10, t1_p50, t1_p90 = _apply_source_profile_quantiles(
        t1_p10,
        t1_p50,
        t1_p90,
        source_calibration=source_calibration,
        target="t1_us",
        source_profile=str(args.source_profile),
        uncertainty_scale=float(args.source_uncertainty_scale),
    )
    t2_p10, t2_p50, t2_p90 = _apply_source_profile_quantiles(
        t2_p10,
        t2_p50,
        t2_p90,
        source_calibration=source_calibration,
        target="t2_us",
        source_profile=str(args.source_profile),
        uncertainty_scale=float(args.source_uncertainty_scale),
    )

    width_t1 = t1_p90 - t1_p10
    width_t2 = t2_p90 - t2_p10

    feat_ref = np.asarray(bundle["feature_ood_train_scaled"], dtype=np.float32)
    feat_thr = float(bundle["feature_ood_threshold"])
    feat_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    feat_nn.fit(feat_ref)
    feat_dist, _ = feat_nn.kneighbors(x_scaled, n_neighbors=1)
    feat_dist = feat_dist[:, 0]
    feature_ood = feat_dist > feat_thr

    emb_dist = np.full(len(candidates), np.nan, dtype=float)
    embedding_ood = np.zeros(len(candidates), dtype=bool)

    emb_ref = bundle.get("embedding_ref", {"enabled": False})
    if bool(emb_ref.get("enabled", False)):
        geom_cols = list(emb_ref["geometry_cols"])
        cfg = emb_ref["model_config"]
        model = TwinEncoder(
            geom_in=int(cfg["geom_in"]),
            phys_in=int(cfg["phys_in"]),
            hidden_dim=int(cfg["hidden_dim"]),
            emb_dim=int(cfg["emb_dim"]),
            dropout=float(cfg["dropout"]),
        )
        model.load_state_dict(emb_ref["state_dict"])
        model.eval()

        g_mean = np.asarray(emb_ref["geom_scaler_mean"], dtype=np.float32)
        g_scale = np.asarray(emb_ref["geom_scaler_scale"], dtype=np.float32)
        z_train = np.asarray(emb_ref["train_embeddings"], dtype=np.float32)
        z_thr = float(emb_ref["ood_threshold"])

        g_vals = candidates.loc[:, geom_cols].to_numpy(dtype=float)
        g_scaled = ((g_vals - g_mean) / g_scale).astype(np.float32)
        with torch.no_grad():
            z = model.encode_geom(torch.from_numpy(g_scaled)).cpu().numpy()
        z = l2_normalize(z)

        z_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        z_nn.fit(z_train)
        z_dist, _ = z_nn.kneighbors(z, n_neighbors=1)
        emb_dist = z_dist[:, 0]
        embedding_ood = emb_dist > z_thr

    combined_ood = feature_ood | embedding_ood

    width_ref = dict(bundle.get("width_reference", {}))
    t1_w_p90 = float(width_ref.get("t1_width_p90", np.nanmax(width_t1) if len(width_t1) > 0 else 1.0))
    t2_w_p90 = float(width_ref.get("t2_width_p90", np.nanmax(width_t2) if len(width_t2) > 0 else 1.0))

    conf = np.array(["high"] * len(candidates), dtype=object)
    conf[combined_ood] = "low"
    med_mask = (~combined_ood) & ((width_t1 > t1_w_p90) | (width_t2 > t2_w_p90))
    conf[med_mask] = "medium"

    robust_mean = pd.to_numeric(candidates.get("robust_mean_error"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    robust_std = pd.to_numeric(candidates.get("robust_std_error"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    objective_total = pd.to_numeric(candidates.get("objective_total"), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    pass_t1 = t1_p10 >= float(args.min_t1_p10_us)
    pass_ood = ~combined_ood if not args.allow_ood else np.ones(len(candidates), dtype=bool)
    pass_conf = conf != "low" if not args.allow_low_confidence else np.ones(len(candidates), dtype=bool)
    pass_unc = (
        (width_t1 <= t1_w_p90 * float(args.max_t1_width_multiplier))
        & (width_t2 <= t2_w_p90 * float(args.max_t2_width_multiplier))
    )
    if args.allow_uncertain:
        pass_unc = np.ones(len(candidates), dtype=bool)

    passed = pass_t1 & pass_ood & pass_conf & pass_unc

    score = (
        float(args.score_w_t1_p10) * t1_p10
        - float(args.score_w_t1_width) * width_t1
        - float(args.score_w_objective) * objective_total
        - float(args.score_w_robust) * (robust_mean + float(args.robust_std_weight) * robust_std)
        - float(args.score_w_ood) * combined_ood.astype(float)
        - float(args.score_w_low_confidence) * (conf == "low").astype(float)
    )

    out = candidates.copy()
    out["coh_t1_p10_us"] = t1_p10
    out["coh_t1_p50_us"] = t1_p50
    out["coh_t1_p90_us"] = t1_p90
    out["coh_t2_p10_us"] = t2_p10
    out["coh_t2_p50_us"] = t2_p50
    out["coh_t2_p90_us"] = t2_p90
    out["coh_t1_width_us"] = width_t1
    out["coh_t2_width_us"] = width_t2
    out["coh_feature_ood"] = feature_ood
    out["coh_embedding_ood"] = embedding_ood
    out["coh_combined_ood"] = combined_ood
    out["coh_feature_distance"] = feat_dist
    out["coh_embedding_distance"] = emb_dist
    out["coh_confidence"] = conf
    out["pass_t1"] = pass_t1
    out["pass_ood"] = pass_ood
    out["pass_confidence"] = pass_conf
    out["pass_uncertainty"] = pass_unc
    out["pass_all"] = passed
    out["phase5_score"] = score

    out = out.sort_values(["pass_all", "phase5_score", "coh_t1_p10_us"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def write_report(outdir: Path, scored: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace, targets: pd.DataFrame) -> None:
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "targets_total": int(len(targets)),
        "candidates_total": int(len(scored)),
        "candidates_pass_all": int(scored["pass_all"].sum()) if len(scored) > 0 else 0,
        "selected_total": int(len(selected)),
        "config": {
            "min_t1_p10_us": float(args.min_t1_p10_us),
            "max_t1_width_multiplier": float(args.max_t1_width_multiplier),
            "max_t2_width_multiplier": float(args.max_t2_width_multiplier),
            "allow_ood": bool(args.allow_ood),
            "allow_low_confidence": bool(args.allow_low_confidence),
            "allow_uncertain": bool(args.allow_uncertain),
            "source_profile": str(args.source_profile),
            "source_uncertainty_scale": float(args.source_uncertainty_scale),
            "selected_top_n": int(args.selected_top_n),
            "fallback_top_n": int(args.fallback_top_n),
        },
        "metrics": {
            "selected_t1_p10_median": float(selected["coh_t1_p10_us"].median()) if len(selected) > 0 else None,
            "selected_t1_p10_min": float(selected["coh_t1_p10_us"].min()) if len(selected) > 0 else None,
            "selected_objective_median": float(selected["objective_total"].median()) if len(selected) > 0 else None,
            "selected_ood_rate": float(selected["coh_combined_ood"].mean()) if len(selected) > 0 else None,
        },
    }

    (outdir / "phase5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# Phase 5 Closed-Loop Candidate Report")
    md.append("")
    md.append(f"Generated: {summary['generated_at_utc']}")
    md.append("")
    md.append("## Counts")
    md.append("")
    md.append(f"- Targets: {summary['targets_total']}")
    md.append(f"- Candidates: {summary['candidates_total']}")
    md.append(f"- Pass-all candidates: {summary['candidates_pass_all']}")
    md.append(f"- Selected: {summary['selected_total']}")
    md.append("")
    md.append("## Selection Quality")
    md.append("")
    md.append(f"- Selected median t1_p10 (us): {summary['metrics']['selected_t1_p10_median']}")
    md.append(f"- Selected min t1_p10 (us): {summary['metrics']['selected_t1_p10_min']}")
    md.append(f"- Selected median objective_total: {summary['metrics']['selected_objective_median']}")
    md.append(f"- Selected OOD rate: {summary['metrics']['selected_ood_rate']}")
    md.append("")
    md.append("## Top Selected")
    md.append("")
    show = selected.head(15)
    if len(show) == 0:
        md.append("No candidates selected.")
    else:
        for _, r in show.iterrows():
            md.append(
                "- "
                f"target={r.get('target_id')} rank={int(r.get('rank', 0))} score={float(r.get('phase5_score')):.6f} "
                f"t1_p10={float(r.get('coh_t1_p10_us')):.6f}us ood={bool(r.get('coh_combined_ood'))} "
                f"objective={float(r.get('objective_total')):.6f}"
            )

    (outdir / "phase5_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args)
    targets.to_csv(outdir / "phase5_targets_used.csv", index=False)

    phase3_df = run_phase3_targets(args, targets)
    if phase3_df.empty:
        raise SystemExit("Phase 5 produced no Phase 3 candidates")

    scored = phase4_score_candidates(args, phase3_df)
    scored.to_csv(outdir / "phase5_candidate_batch.csv", index=False)

    selected = scored[scored["pass_all"]].copy()
    if len(selected) == 0:
        selected = scored.head(max(1, int(args.fallback_top_n))).copy()
    else:
        selected = selected.head(max(1, int(args.selected_top_n))).copy()
    selected.to_csv(outdir / "phase5_selected_candidates.csv", index=False)

    write_report(outdir, scored, selected, args, targets)

    print("=== Phase 5 Closed-Loop Complete ===")
    print(f"targets={len(targets)} candidates={len(scored)} pass_all={int(scored['pass_all'].sum())} selected={len(selected)}")
    print(f"candidate_batch={outdir / 'phase5_candidate_batch.csv'}")
    print(f"selected={outdir / 'phase5_selected_candidates.csv'}")
    print(f"report={outdir / 'phase5_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())