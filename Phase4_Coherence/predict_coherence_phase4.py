#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
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


def predict_target_quantiles(models: Dict[str, object], target: str, x: np.ndarray, transform: Dict[str, float]) -> Tuple[float, float, float]:
    q10_m = models[f"{target}_q10"].predict(x)
    q50_m = models[f"{target}_q50"].predict(x)
    q90_m = models[f"{target}_q90"].predict(x)
    q10 = float(invert_target_transform(q10_m, transform)[0])
    q50 = float(invert_target_transform(q50_m, transform)[0])
    q90 = float(invert_target_transform(q90_m, transform)[0])
    arr = np.sort(np.array([q10, q50, q90], dtype=float))
    return float(arr[0]), float(arr[1]), float(arr[2])


def apply_source_profile_scalar(
    q10: float,
    q50: float,
    q90: float,
    source_calibration: Dict[str, object],
    target: str,
    source_profile: str,
    uncertainty_scale: float,
) -> Tuple[float, float, float]:
    arr = np.array([q10, q50, q90], dtype=float)

    targets = source_calibration.get("targets", {}) if isinstance(source_calibration, dict) else {}
    tcal = targets.get(target, {}) if isinstance(targets, dict) else {}
    offsets = tcal.get("offsets_log10", {}) if isinstance(tcal, dict) else {}

    offset = 0.0
    if source_profile and source_profile != "global" and isinstance(offsets, dict):
        offset = float(offsets.get(source_profile, 0.0))
    if abs(offset) > 0:
        arr = arr * (10.0 ** offset)

    if source_profile == "global" and isinstance(tcal, dict):
        spread = max(0.0, float(tcal.get("offset_std_log10", 0.0)) * max(0.0, float(uncertainty_scale)))
        if spread > 0:
            factor = 10.0 ** spread
            arr[0] = min(arr[0], arr[1] / factor)
            arr[2] = max(arr[2], arr[1] * factor)

    arr = np.sort(arr)
    return float(arr[0]), float(arr[1]), float(arr[2])


def apply_quantile_inflation_scalar(
    q10: float,
    q50: float,
    q90: float,
    factor: float,
) -> Tuple[float, float, float]:
    f = float(max(1.0, factor))
    lo = max(float(q50) - float(q10), 1e-18)
    hi = max(float(q90) - float(q50), 1e-18)
    arr = np.array([q50 - lo * f, q50, q50 + hi * f], dtype=float)
    arr.sort()
    return float(arr[0]), float(arr[1]), float(arr[2])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Predict coherence with Phase 4 model")
    parser.add_argument("--bundle-path", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "phase4_coherence_bundle.joblib")
    parser.add_argument("--phase1-model-path", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-metadata-path", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")

    parser.add_argument("--candidate-json", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=1)

    parser.add_argument("--pad-width-um", type=float, default=None)
    parser.add_argument("--pad-height-um", type=float, default=None)
    parser.add_argument("--gap-um", type=float, default=None)
    parser.add_argument("--junction-area-um2", type=float, default=None)

    parser.add_argument("--freq-01-ghz", type=float, default=None)
    parser.add_argument("--anharmonicity-ghz", type=float, default=None)
    parser.add_argument("--ej-ghz", type=float, default=None)
    parser.add_argument("--ec-ghz", type=float, default=None)
    parser.add_argument("--charge-sensitivity-ghz", type=float, default=None)
    parser.add_argument("--ej-ec-ratio", type=float, default=None)

    parser.add_argument("--disable-phase1-fill", action="store_true")
    parser.add_argument("--source-profile", type=str, default="global")
    parser.add_argument("--source-uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def read_candidate(candidate_json: Path, rank: int) -> Dict[str, float]:
    payload = json.loads(candidate_json.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not rows:
        return {}

    idx = max(1, rank) - 1
    idx = min(idx, len(rows) - 1)
    row = rows[idx]

    out: Dict[str, float] = {}
    direct = {
        "pad_width_um": "pad_width_um",
        "pad_height_um": "pad_height_um",
        "gap_um": "gap_um",
        "junction_area_um2": "junction_area_um2",
        "freq_01_GHz": "freq_01_GHz",
        "anharmonicity_GHz": "anharmonicity_GHz",
        "EJ_GHz": "EJ_GHz",
        "EC_GHz": "EC_GHz",
        "charge_sensitivity_GHz": "charge_sensitivity_GHz",
        "EJ_EC_ratio": "EJ_EC_ratio",
    }
    pred = {
        "freq_01_GHz": "pred_freq_01_GHz",
        "anharmonicity_GHz": "pred_anharmonicity_GHz",
        "EJ_GHz": "pred_EJ_GHz",
        "EC_GHz": "pred_EC_GHz",
        "charge_sensitivity_GHz": "pred_charge_sensitivity_GHz",
    }

    for out_key, in_key in direct.items():
        if in_key in row and row[in_key] is not None:
            out[out_key] = float(row[in_key])

    for out_key, in_key in pred.items():
        if out_key not in out and in_key in row and row[in_key] is not None:
            out[out_key] = float(row[in_key])

    return out


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)

    feature_cols: List[str] = list(bundle["feature_cols"])
    target_transforms: Dict[str, Dict[str, float]] = dict(bundle["target_transforms"])
    feature_medians: Dict[str, float] = {k: float(v) for k, v in dict(bundle["feature_medians"]).items()}

    values: Dict[str, Optional[float]] = {c: None for c in feature_cols}
    notes: List[str] = []

    if args.candidate_json is not None and args.candidate_json.exists():
        cand = read_candidate(args.candidate_json, args.rank)
        for k, v in cand.items():
            if k in values:
                values[k] = float(v)
        notes.append(f"Loaded candidate from {args.candidate_json} rank={args.rank}")

    cli_map = {
        "pad_width_um": args.pad_width_um,
        "pad_height_um": args.pad_height_um,
        "gap_um": args.gap_um,
        "junction_area_um2": args.junction_area_um2,
        "freq_01_GHz": args.freq_01_ghz,
        "anharmonicity_GHz": args.anharmonicity_ghz,
        "EJ_GHz": args.ej_ghz,
        "EC_GHz": args.ec_ghz,
        "charge_sensitivity_GHz": args.charge_sensitivity_ghz,
        "EJ_EC_ratio": args.ej_ec_ratio,
    }
    for k, v in cli_map.items():
        if v is not None and k in values:
            values[k] = float(v)

    geom_required = ["pad_width_um", "pad_height_um", "gap_um", "junction_area_um2"]
    missing_geom = [c for c in geom_required if c in values and values[c] is None]
    if missing_geom:
        raise SystemExit(f"Missing geometry inputs: {missing_geom}")

    phase1_filled = False
    phase1_keys = ["freq_01_GHz", "anharmonicity_GHz", "EJ_GHz", "EC_GHz", "charge_sensitivity_GHz"]
    if (not args.disable_phase1_fill) and any((k in values and values[k] is None) for k in phase1_keys):
        if args.phase1_model_path.exists() and args.phase1_metadata_path.exists():
            phase1_model = joblib.load(args.phase1_model_path)
            phase1_meta = json.loads(args.phase1_metadata_path.read_text(encoding="utf-8"))
            feature_order = phase1_meta["feature_cols"]
            target_order = phase1_meta["target_cols"]

            row = {
                "pad_width_um": float(values.get("pad_width_um")),
                "pad_height_um": float(values.get("pad_height_um")),
                "gap_um": float(values.get("gap_um")),
                "junction_area_um2": float(values.get("junction_area_um2")),
            }
            x = np.array([[row[c] for c in feature_order]], dtype=float)
            pred = phase1_model.predict(x).reshape(-1)
            pred_map = {target_order[i]: float(pred[i]) for i in range(len(target_order))}
            for k in phase1_keys:
                if k in values and values[k] is None and k in pred_map:
                    values[k] = pred_map[k]
            phase1_filled = True
            notes.append("Filled missing physics features from Phase1 surrogate")

    if "EJ_EC_ratio" in values and values["EJ_EC_ratio"] is None and values.get("EJ_GHz") is not None and values.get("EC_GHz") is not None:
        ec = float(values["EC_GHz"])
        if abs(ec) > 1e-12:
            values["EJ_EC_ratio"] = float(float(values["EJ_GHz"]) / ec)

    imputed: List[str] = []
    for c in feature_cols:
        if values[c] is None:
            values[c] = feature_medians[c]
            imputed.append(c)
    if imputed:
        notes.append(f"Imputed features with train medians: {imputed}")

    x_raw = np.array([[float(values[c]) for c in feature_cols]], dtype=float)
    x_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
    x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

    models = bundle["models"]
    t1_p10, t1_p50, t1_p90 = predict_target_quantiles(models, "t1_us", x_scaled, target_transforms["t1_us"])
    t2_p10, t2_p50, t2_p90 = predict_target_quantiles(models, "t2_us", x_scaled, target_transforms["t2_us"])

    source_calibration = bundle.get("source_calibration", {})
    t1_p10, t1_p50, t1_p90 = apply_source_profile_scalar(
        t1_p10,
        t1_p50,
        t1_p90,
        source_calibration=source_calibration,
        target="t1_us",
        source_profile=str(args.source_profile),
        uncertainty_scale=float(args.source_uncertainty_scale),
    )
    t2_p10, t2_p50, t2_p90 = apply_source_profile_scalar(
        t2_p10,
        t2_p50,
        t2_p90,
        source_calibration=source_calibration,
        target="t2_us",
        source_profile=str(args.source_profile),
        uncertainty_scale=float(args.source_uncertainty_scale),
    )

    anchor_distance = None
    inflation_factor = 1.0
    unc_cfg = bundle.get("uncertainty_inflation_config", {})
    measured_anchor_ref = np.asarray(bundle.get("measured_anchor_feature_scaled", np.empty((0, len(feature_cols)))), dtype=np.float32)
    if measured_anchor_ref.ndim == 2 and measured_anchor_ref.shape[0] > 0 and measured_anchor_ref.shape[1] == x_scaled.shape[1]:
        nn_anchor = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn_anchor.fit(measured_anchor_ref)
        dist_anchor, _ = nn_anchor.kneighbors(x_scaled, n_neighbors=1)
        anchor_distance = float(dist_anchor[0, 0])
        dist_scale = float(max(float(unc_cfg.get("distance_scale", 1.0)), 1e-6))
        dist_gain = float(max(0.0, float(unc_cfg.get("distance_gain", 0.0))))
        max_factor = float(max(1.0, float(unc_cfg.get("max_factor", 2.0))))
        inflation_factor = 1.0 + dist_gain * min(anchor_distance / dist_scale, 3.0)
        inflation_factor = float(np.clip(inflation_factor, 1.0, max_factor))

    t1_p10, t1_p50, t1_p90 = apply_quantile_inflation_scalar(t1_p10, t1_p50, t1_p90, inflation_factor)
    t2_p10, t2_p50, t2_p90 = apply_quantile_inflation_scalar(t2_p10, t2_p50, t2_p90, inflation_factor)

    feat_ref = np.asarray(bundle["feature_ood_train_scaled"], dtype=np.float32)
    feat_thr = float(bundle["feature_ood_threshold"])
    feat_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    feat_nn.fit(feat_ref)
    feat_dist, _ = feat_nn.kneighbors(x_scaled, n_neighbors=1)
    feature_dist = float(feat_dist[0, 0])
    feature_ood = bool(feature_dist > feat_thr)

    embedding_dist = None
    embedding_ood = False
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

        geom_row = np.array([[float(values[c]) for c in geom_cols]], dtype=float)
        geom_scaled = ((geom_row - g_mean) / g_scale).astype(np.float32)
        z = l2_normalize(model.encode_geom(torch.from_numpy(geom_scaled)).detach().cpu().numpy())

        z_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        z_nn.fit(z_train)
        z_dist, _ = z_nn.kneighbors(z, n_neighbors=1)
        embedding_dist = float(z_dist[0, 0])
        embedding_ood = bool(embedding_dist > z_thr)

    combined_ood = bool(feature_ood or embedding_ood)

    width_ref = bundle.get("width_reference", {})
    width_t1 = float(t1_p90 - t1_p10)
    width_t2 = float(t2_p90 - t2_p10)

    confidence = "high"
    if combined_ood:
        confidence = "low"
    elif width_t1 > float(width_ref.get("t1_width_p90", width_t1 + 1.0)) or width_t2 > float(width_ref.get("t2_width_p90", width_t2 + 1.0)):
        confidence = "medium"

    payload = {
        "input_features": {c: float(values[c]) for c in feature_cols},
        "predictions": {
            "t1_us": {
                "p10": t1_p10,
                "p50": t1_p50,
                "p90": t1_p90,
                "interval_width": width_t1,
            },
            "t2_us": {
                "p10": t2_p10,
                "p50": t2_p50,
                "p90": t2_p90,
                "interval_width": width_t2,
            },
        },
        "ood": {
            "feature_distance": feature_dist,
            "feature_threshold": feat_thr,
            "feature_ood": feature_ood,
            "embedding_distance": embedding_dist,
            "embedding_threshold": float(emb_ref.get("ood_threshold", np.nan)) if bool(emb_ref.get("enabled", False)) else None,
            "embedding_ood": embedding_ood,
            "combined_ood": combined_ood,
            "measured_anchor_feature_distance": anchor_distance,
            "uncertainty_inflation_factor": inflation_factor,
        },
        "confidence": confidence,
        "phase1_fill_used": phase1_filled,
        "source_profile": str(args.source_profile),
        "source_uncertainty_scale": float(args.source_uncertainty_scale),
        "notes": notes,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
