
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

FEATURE_COLS: Tuple[str, ...] = (
    "pad_width_um",
    "pad_height_um",
    "gap_um",
    "junction_area_um2",
    "freq_01_GHz",
    "anharmonicity_GHz",
    "EJ_GHz",
    "EC_GHz",
    "charge_sensitivity_GHz",
    "EJ_EC_ratio",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    preferred_measurement = root / "Dataset" / "measurement_dataset_public_bootstrap_augmented.csv"
    default_measurement = preferred_measurement if preferred_measurement.exists() else (root / "Dataset" / "measurement_dataset_public_bootstrap.csv")
    parser = argparse.ArgumentParser(description="Phase 7 evidence hardening and interface packaging")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument(
        "--measurement-csv",
        type=Path,
        default=default_measurement,
    )
    parser.add_argument(
        "--phase6-script",
        type=Path,
        default=root / "Phase6_Reliability" / "run_phase6_reliability.py",
    )
    parser.add_argument(
        "--phase4-train-script",
        type=Path,
        default=root / "Phase4_Coherence" / "train_phase4_coherence.py",
    )
    parser.add_argument(
        "--phase5-script",
        type=Path,
        default=root / "Phase5_ClosedLoop" / "run_phase5_closed_loop.py",
    )
    parser.add_argument(
        "--phase6-summary",
        type=Path,
        default=root / "Phase6_Reliability" / "artifacts" / "phase6_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))

    parser.add_argument("--seeds", type=str, default="42,123,777")
    parser.add_argument("--holdout-min-rows", type=int, default=4)
    parser.add_argument("--holdout-min-sources", type=int, default=2)
    parser.add_argument("--holdout-random-state", type=int, default=42)
    parser.add_argument(
        "--holdout-mode",
        choices=("design_disjoint", "label_only"),
        default="design_disjoint",
        help="design_disjoint excludes holdout IDs from both measurement and single-csv training pools",
    )
    parser.add_argument(
        "--allow-underpowered-holdout",
        action="store_true",
        help="Allow holdout run even when eligible source count is below --holdout-min-sources",
    )

    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    seeds: List[int] = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    return seeds


def run_cmd(cmd: Sequence[str], workdir: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(
        list(cmd),
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-40:])
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nLog: {log_path}\n--- tail ---\n{tail}"
        )


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())


def invert_target_transform(y_model: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    y = y_model.copy()
    if bool(transform["use_log"]):
        y = np.power(10.0, y)
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    return np.clip(y, lo, hi)


def predict_q50(models: Dict[str, object], target: str, x: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    q50 = models[f"{target}_q50"].predict(x)
    return invert_target_transform(q50, transform)


def _source_offset_log10(bundle: Dict[str, object], target: str, source_name: str) -> float:
    source_calibration = bundle.get("source_calibration", {})
    if not isinstance(source_calibration, dict):
        return 0.0
    targets = source_calibration.get("targets", {})
    if not isinstance(targets, dict):
        return 0.0
    tcal = targets.get(target, {})
    if not isinstance(tcal, dict):
        return 0.0
    offsets = tcal.get("offsets_log10", {})
    if not isinstance(offsets, dict):
        return 0.0
    if not source_name:
        return 0.0
    return float(offsets.get(str(source_name), 0.0))


def run_multiseed_phase6(root: Path, args: argparse.Namespace, seeds: Sequence[int], outdir: Path) -> pd.DataFrame:
    runs_dir = outdir / "phase6_multiseed"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = runs_dir / f"seed_{seed}"
        cmd = [
            str(args.python_bin),
            str(args.phase6_script),
            "--skip-phase5",
            "--random-state",
            str(seed),
            "--single-csv",
            str(args.single_csv),
            "--measurement-csv",
            str(args.measurement_csv),
            "--output-dir",
            str(seed_dir),
        ]
        run_cmd(cmd, workdir=root, log_path=seed_dir / "phase6_multiseed.log")

        summary_csv = seed_dir / "phase6_phase4_variant_summary.csv"
        if not summary_csv.exists():
            raise RuntimeError(f"Missing Phase 6 summary for seed {seed}: {summary_csv}")

        sdf = pd.read_csv(summary_csv)
        sdf.insert(0, "seed", int(seed))
        rows.append(sdf)

    all_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_df.to_csv(outdir / "phase7_multiseed_raw.csv", index=False)
    return all_df


def summarize_multiseed(multiseed_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    if multiseed_df.empty:
        return pd.DataFrame()

    numeric_metrics = [
        "reliability_rank",
        "ood_t1_mae",
        "ood_t2_log10_mae",
        "cal_gap_ood_t1",
        "cal_gap_ood_t2",
        "eval_all_t1_us_aurc",
        "eval_all_t2_us_log10_aurc",
        "eval_all_t1_us_abstain20_gain",
        "eval_all_t2_us_log10_abstain20_gain",
    ]
    present = [c for c in numeric_metrics if c in multiseed_df.columns]

    agg = (
        multiseed_df.groupby("variant", as_index=False)[present]
        .agg(["mean", "std", "min", "max"])  # type: ignore[arg-type]
    )
    agg.columns = ["variant"] + [f"{c[0]}_{c[1]}" for c in agg.columns[1:]]

    top1_counts = (
        multiseed_df.loc[multiseed_df["reliability_rank"] == 1, "variant"]
        .value_counts()
        .rename_axis("variant")
        .reset_index(name="top1_count")
    )
    agg = agg.merge(top1_counts, on="variant", how="left")
    agg["top1_count"] = agg["top1_count"].fillna(0).astype(int)

    agg = agg.sort_values(["reliability_rank_mean", "ood_t1_mae_mean"], ascending=[True, True]).reset_index(drop=True)
    agg.to_csv(outdir / "phase7_multiseed_summary.csv", index=False)

    return agg


def evaluate_bundle_on_holdout(
    bundle_path: Path,
    holdout_rows: pd.DataFrame,
    single_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bundle = joblib.load(bundle_path)
    feature_cols = list(bundle["feature_cols"])
    transforms = dict(bundle["target_transforms"])
    models = bundle["models"]

    rows: List[Dict[str, float]] = []

    for _, r in holdout_rows.iterrows():
        row_idx = pd.to_numeric(r.get("row_index"), errors="coerce")
        if not np.isfinite(row_idx):
            continue
        row_idx_i = int(row_idx)
        if row_idx_i < 0 or row_idx_i >= len(single_df):
            continue

        src = single_df.iloc[row_idx_i]
        feat = []
        valid = True
        for c in feature_cols:
            v = pd.to_numeric(src.get(c), errors="coerce")
            if not np.isfinite(v):
                valid = False
                break
            feat.append(float(v))
        if not valid:
            continue

        x_raw = np.asarray(feat, dtype=float).reshape(1, -1)
        x_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
        x_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
        x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

        pred_t1 = float(predict_q50(models, "t1_us", x_scaled, transforms["t1_us"])[0])
        pred_t2 = float(predict_q50(models, "t2_us", x_scaled, transforms["t2_us"])[0])

        m_t1 = pd.to_numeric(r.get("measured_t1_us"), errors="coerce")
        m_t2 = pd.to_numeric(r.get("measured_t2_us"), errors="coerce")
        source_name = str(r.get("source_name", "") or "")

        t1_off = _source_offset_log10(bundle, "t1_us", source_name)
        t2_off = _source_offset_log10(bundle, "t2_us", source_name)
        m_t1_c = (float(m_t1) / (10.0 ** t1_off)) if np.isfinite(m_t1) and float(m_t1) > 0 else np.nan
        m_t2_c = (float(m_t2) / (10.0 ** t2_off)) if np.isfinite(m_t2) and float(m_t2) > 0 else np.nan

        rec: Dict[str, float] = {
            "row_index": float(row_idx_i),
            "pred_t1_us": pred_t1,
            "pred_t2_us": pred_t2,
            "measured_t1_us": float(m_t1) if np.isfinite(m_t1) else np.nan,
            "measured_t2_us": float(m_t2) if np.isfinite(m_t2) else np.nan,
            "measured_t1_canonical_us": m_t1_c,
            "measured_t2_canonical_us": m_t2_c,
        }
        if np.isfinite(m_t1_c):
            rec["abs_err_t1_us"] = abs(m_t1_c - pred_t1)
        else:
            rec["abs_err_t1_us"] = np.nan

        if np.isfinite(m_t2_c):
            y = max(m_t2_c, 1e-18)
            p = max(pred_t2, 1e-18)
            rec["abs_err_t2_log10"] = abs(np.log10(y) - np.log10(p))
        else:
            rec["abs_err_t2_log10"] = np.nan

        rows.append(rec)

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return pred_df, {
            "holdout_rows_used": 0,
            "holdout_t1_count": 0,
            "holdout_t2_count": 0,
            "holdout_t1_mae": np.nan,
            "holdout_t2_log10_mae": np.nan,
            "holdout_unique_designs_used": 0,
            "holdout_t1_count_unique_design": 0,
            "holdout_t2_count_unique_design": 0,
            "holdout_t1_mae_unique_design": np.nan,
            "holdout_t2_log10_mae_unique_design": np.nan,
        }

    t1_mask = pred_df["abs_err_t1_us"].notna()
    t2_mask = pred_df["abs_err_t2_log10"].notna()
    by_design = pred_df.groupby("row_index", as_index=False).agg(
        pred_t1_us=("pred_t1_us", "mean"),
        pred_t2_us=("pred_t2_us", "mean"),
        measured_t1_canonical_us=("measured_t1_canonical_us", "mean"),
        measured_t2_canonical_us=("measured_t2_canonical_us", "mean"),
    )
    by_design["abs_err_t1_us"] = np.where(
        np.isfinite(by_design["measured_t1_canonical_us"]),
        np.abs(by_design["measured_t1_canonical_us"] - by_design["pred_t1_us"]),
        np.nan,
    )
    by_design["abs_err_t2_log10"] = np.where(
        np.isfinite(by_design["measured_t2_canonical_us"]),
        np.abs(
            np.log10(np.maximum(by_design["measured_t2_canonical_us"], 1e-18))
            - np.log10(np.maximum(by_design["pred_t2_us"], 1e-18))
        ),
        np.nan,
    )
    t1_design_mask = by_design["abs_err_t1_us"].notna()
    t2_design_mask = by_design["abs_err_t2_log10"].notna()

    metrics = {
        "holdout_rows_used": int(len(pred_df)),
        "holdout_t1_count": int(t1_mask.sum()),
        "holdout_t2_count": int(t2_mask.sum()),
        "holdout_t1_mae": float(pred_df.loc[t1_mask, "abs_err_t1_us"].mean()) if int(t1_mask.sum()) > 0 else np.nan,
        "holdout_t2_log10_mae": float(pred_df.loc[t2_mask, "abs_err_t2_log10"].mean()) if int(t2_mask.sum()) > 0 else np.nan,
        "holdout_unique_designs_used": int(len(by_design)),
        "holdout_t1_count_unique_design": int(t1_design_mask.sum()),
        "holdout_t2_count_unique_design": int(t2_design_mask.sum()),
        "holdout_t1_mae_unique_design": float(by_design.loc[t1_design_mask, "abs_err_t1_us"].mean()) if int(t1_design_mask.sum()) > 0 else np.nan,
        "holdout_t2_log10_mae_unique_design": float(by_design.loc[t2_design_mask, "abs_err_t2_log10"].mean()) if int(t2_design_mask.sum()) > 0 else np.nan,
    }
    return pred_df, metrics


def _extract_id_set(df: pd.DataFrame) -> Tuple[Optional[str], np.ndarray]:
    if "row_index" in df.columns:
        v = pd.to_numeric(df["row_index"], errors="coerce")
        if np.isfinite(v).any():
            return "row_index", v.dropna().astype(int).unique()
    if "design_id" in df.columns:
        v = pd.to_numeric(df["design_id"], errors="coerce")
        if np.isfinite(v).any():
            return "design_id", v.dropna().astype(int).unique()
    return None, np.asarray([], dtype=int)


def _unique_design_count(df: pd.DataFrame) -> Tuple[int, Optional[str]]:
    id_col, ids = _extract_id_set(df)
    return int(len(ids)), id_col


def run_source_holdout(root: Path, args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    holdout_dir = outdir / "source_holdout"
    runs_dir = holdout_dir / "runs"
    data_dir = holdout_dir / "datasets"
    audit_dir = holdout_dir / "audits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    measurement = pd.read_csv(args.measurement_csv)
    if "source_name" not in measurement.columns:
        raise RuntimeError("measurement CSV missing source_name column for holdout")

    if "is_synthetic_regularization" in measurement.columns:
        synth_mask = measurement["is_synthetic_regularization"].fillna(False)
        if synth_mask.dtype != bool:
            synth_txt = synth_mask.astype(str).str.strip().str.lower()
            synth_mask = synth_txt.isin({"1", "true", "t", "yes", "y"})
        synth_mask = synth_mask.astype(bool)
    else:
        synth_mask = pd.Series(np.zeros(len(measurement), dtype=bool), index=measurement.index)

    measurement_direct = measurement.loc[~synth_mask].copy()
    source_counts = measurement_direct["source_name"].fillna("<missing>").value_counts()
    keep_sources = [str(s) for s, n in source_counts.items() if int(n) >= int(args.holdout_min_rows)]
    n_unique_designs_total, id_col_total = _unique_design_count(measurement_direct)
    eligible_mask = measurement_direct["source_name"].fillna("<missing>").isin(set(keep_sources))
    n_unique_designs_eligible, id_col_eligible = _unique_design_count(measurement_direct.loc[eligible_mask].copy())

    if not keep_sources:
        raise RuntimeError("No source has enough rows for holdout evaluation")
    if len(keep_sources) < int(args.holdout_min_sources) and not bool(args.allow_underpowered_holdout):
        raise RuntimeError(
            f"Underpowered holdout: eligible_sources={len(keep_sources)} < holdout_min_sources={int(args.holdout_min_sources)}. "
            "Use --allow-underpowered-holdout to override."
        )

    single_df = pd.read_csv(args.single_csv)
    rows: List[Dict[str, object]] = []
    leak_rows: List[Dict[str, object]] = []

    for source in keep_sources:
        src_slug = slugify(source)
        source_mask = measurement["source_name"].fillna("<missing>") == source

        test_df = measurement[source_mask].copy()
        train_df = measurement[~source_mask].copy()
        anchor_excluded_rows = 0
        if "anchor_source_name" in train_df.columns:
            anchor_excluded_rows = int(np.sum(train_df["anchor_source_name"].fillna("<missing>").astype(str) == source))
            train_df = train_df[train_df["anchor_source_name"].fillna("<missing>").astype(str) != source].copy()

        holdout_id_col, holdout_ids = _extract_id_set(test_df)
        train_single_df = single_df.copy()
        if args.holdout_mode == "design_disjoint" and holdout_id_col is not None and holdout_ids.size > 0:
            if holdout_id_col == "row_index":
                train_single_df = train_single_df.reset_index().rename(columns={"index": "row_index"})
                train_single_df = train_single_df[~train_single_df["row_index"].isin(set(holdout_ids.tolist()))].copy()
            elif holdout_id_col == "design_id" and "design_id" in train_single_df.columns:
                train_single_df = train_single_df[~pd.to_numeric(train_single_df["design_id"], errors="coerce").isin(set(holdout_ids.tolist()))].copy()

            if holdout_id_col in train_df.columns:
                train_df = train_df[~pd.to_numeric(train_df[holdout_id_col], errors="coerce").isin(set(holdout_ids.tolist()))].copy()

        train_id_col, train_ids = _extract_id_set(train_df)
        single_train_id_col, _ = _extract_id_set(train_single_df)

        overlap_train_measure = int(len(set(holdout_ids.tolist()) & set(train_ids.tolist()))) if holdout_ids.size > 0 else 0
        overlap_train_single = 0
        if holdout_ids.size > 0:
            if holdout_id_col == "row_index" and "row_index" in train_single_df.columns:
                overlap_train_single = int(len(set(holdout_ids.tolist()) & set(pd.to_numeric(train_single_df["row_index"], errors="coerce").dropna().astype(int).tolist())))
            elif holdout_id_col == "design_id" and "design_id" in train_single_df.columns:
                overlap_train_single = int(len(set(holdout_ids.tolist()) & set(pd.to_numeric(train_single_df["design_id"], errors="coerce").dropna().astype(int).tolist())))

        train_csv = data_dir / f"train_without__{src_slug}.csv"
        test_csv = data_dir / f"test_only__{src_slug}.csv"
        train_single_csv = data_dir / f"train_single_without__{src_slug}.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        train_single_df.to_csv(train_single_csv, index=False)

        leak_rows.append(
            {
                "source_name": source,
                "holdout_mode": str(args.holdout_mode),
                "holdout_id_col": holdout_id_col,
                "holdout_ids_count": int(len(holdout_ids)),
                "anchor_linked_rows_excluded": int(anchor_excluded_rows),
                "train_measurement_id_col": train_id_col,
                "train_measurement_overlap_count": int(overlap_train_measure),
                "train_single_id_col": holdout_id_col if holdout_id_col in ("row_index", "design_id") else single_train_id_col,
                "train_single_overlap_count": int(overlap_train_single),
                "overlap_count": int(max(overlap_train_measure, overlap_train_single)),
            }
        )
        if args.holdout_mode == "design_disjoint" and (overlap_train_measure > 0 or overlap_train_single > 0):
            raise RuntimeError(
                f"Leakage audit failed for source={source}: measurement_overlap={overlap_train_measure}, "
                f"single_overlap={overlap_train_single}"
            )

        run_dir = runs_dir / f"holdout__{src_slug}"
        cmd = [
            str(args.python_bin),
            str(args.phase4_train_script),
            "--single-csv",
            str(train_single_csv if args.holdout_mode == "design_disjoint" else args.single_csv),
            "--measurement-csv",
            str(train_csv),
            "--label-mode",
            "hybrid",
            "--measured-weight",
            "3.0",
            "--proxy-weight",
            "1.0",
            "--output-dir",
            str(run_dir),
            "--random-state",
            str(args.holdout_random_state),
        ]
        run_cmd(cmd, workdir=root, log_path=run_dir / "train.log")

        bundle_path = run_dir / "phase4_coherence_bundle.joblib"
        pred_df, metrics = evaluate_bundle_on_holdout(bundle_path=bundle_path, holdout_rows=test_df, single_df=single_df)
        pred_path = run_dir / "holdout_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        rows.append(
            {
                "source_name": source,
                "holdout_mode": str(args.holdout_mode),
                "n_sources_total": int(len(source_counts)),
                "n_sources_eligible": int(len(keep_sources)),
                "n_unique_designs_total": int(n_unique_designs_total),
                "n_unique_designs_eligible_sources": int(n_unique_designs_eligible),
                "n_synthetic_rows_input": int(synth_mask.sum()),
                "eligible_sources_total": int(len(keep_sources)),
                "source_rows": int(len(test_df)),
                "source_rows_with_t1": int(pd.to_numeric(test_df.get("measured_t1_us"), errors="coerce").notna().sum()),
                "source_rows_with_t2": int(pd.to_numeric(test_df.get("measured_t2_us"), errors="coerce").notna().sum()),
                "train_rows_anchor_linked_excluded": int(anchor_excluded_rows),
                "source_unique_designs": int(
                    len(
                        set(
                            (
                                pd.to_numeric(test_df.get("row_index"), errors="coerce")
                                if "row_index" in test_df.columns
                                else pd.to_numeric(test_df.get("design_id"), errors="coerce")
                            ).dropna().astype(int).tolist()
                        )
                    )
                ),
                "n_unique_designs_source": int(
                    len(
                        set(
                            (
                                pd.to_numeric(test_df.get("row_index"), errors="coerce")
                                if "row_index" in test_df.columns
                                else pd.to_numeric(test_df.get("design_id"), errors="coerce")
                            ).dropna().astype(int).tolist()
                        )
                    )
                ),
                "leakage_overlap_measurement": int(overlap_train_measure),
                "leakage_overlap_single": int(overlap_train_single),
                "leakage_overlap_count": int(max(overlap_train_measure, overlap_train_single)),
                **metrics,
                "bundle_path": str(bundle_path.resolve()),
                "predictions_csv": str(pred_path.resolve()),
                "train_single_csv": str(train_single_csv.resolve()),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("source_rows", ascending=False).reset_index(drop=True)
    out_df.to_csv(holdout_dir / "phase7_source_holdout_summary.csv", index=False)

    counts_df = source_counts.rename_axis("source_name").reset_index(name="rows")
    counts_df.to_csv(holdout_dir / "phase7_source_counts.csv", index=False)
    leak_df = pd.DataFrame(leak_rows)
    leak_df.to_csv(audit_dir / "phase7_holdout_leakage_audit.csv", index=False)
    leakage_overlap_max = int(leak_df["overlap_count"].max()) if not leak_df.empty and "overlap_count" in leak_df.columns else 0
    holdout_overview = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_mode": str(args.holdout_mode),
        "n_rows_total_input": int(len(measurement)),
        "n_rows_direct_input": int(len(measurement_direct)),
        "n_rows_synthetic_input": int(synth_mask.sum()),
        "n_sources_total": int(len(source_counts)),
        "n_sources_eligible": int(len(keep_sources)),
        "n_unique_designs_total": int(n_unique_designs_total),
        "n_unique_designs_total_id_col": id_col_total,
        "n_unique_designs_eligible_sources": int(n_unique_designs_eligible),
        "n_unique_designs_eligible_id_col": id_col_eligible,
        "leakage_overlap_max": int(leakage_overlap_max),
        "leakage_overlap_all_zero": bool(leakage_overlap_max == 0),
    }
    (holdout_dir / "phase7_source_holdout_overview.json").write_text(
        json.dumps(holdout_overview, indent=2),
        encoding="utf-8",
    )
    (audit_dir / "phase7_holdout_leakage_audit.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "holdout_mode": str(args.holdout_mode),
                "rows": leak_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_df


def build_paper_outputs(
    outdir: Path,
    multiseed_raw: Optional[pd.DataFrame],
    multiseed_summary: Optional[pd.DataFrame],
    holdout_summary: Optional[pd.DataFrame],
    phase6_summary_path: Path,
) -> None:
    paper_dir = outdir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    phase6_payload = {}
    if phase6_summary_path.exists():
        phase6_payload = json.loads(phase6_summary_path.read_text(encoding="utf-8"))

    top_variant = None
    if multiseed_summary is not None and not multiseed_summary.empty:
        top_variant = str(multiseed_summary.iloc[0]["variant"])

    claim_rows: List[Dict[str, object]] = []
    if multiseed_summary is not None and not multiseed_summary.empty:
        for _, r in multiseed_summary.iterrows():
            claim_rows.append(
                {
                    "section": "multiseed",
                    "variant": r["variant"],
                    "reliability_rank_mean": r.get("reliability_rank_mean"),
                    "reliability_rank_std": r.get("reliability_rank_std"),
                    "ood_t1_mae_mean": r.get("ood_t1_mae_mean"),
                    "ood_t1_mae_std": r.get("ood_t1_mae_std"),
                    "ood_t2_log10_mae_mean": r.get("ood_t2_log10_mae_mean"),
                    "ood_t2_log10_mae_std": r.get("ood_t2_log10_mae_std"),
                    "top1_count": r.get("top1_count"),
                }
            )

    if holdout_summary is not None and not holdout_summary.empty:
        for _, r in holdout_summary.iterrows():
            claim_rows.append(
                {
                    "section": "source_holdout",
                    "variant": "phase4_hybrid_full_conf_excluding_source",
                    "source_name": r.get("source_name"),
                    "n_sources_total": r.get("n_sources_total"),
                    "n_sources_eligible": r.get("n_sources_eligible"),
                    "n_unique_designs_total": r.get("n_unique_designs_total"),
                    "n_unique_designs_eligible_sources": r.get("n_unique_designs_eligible_sources"),
                    "source_rows": r.get("source_rows"),
                    "n_unique_designs_source": r.get("n_unique_designs_source"),
                    "leakage_overlap_count": r.get("leakage_overlap_count"),
                    "holdout_t1_mae": r.get("holdout_t1_mae"),
                    "holdout_t2_log10_mae": r.get("holdout_t2_log10_mae"),
                    "holdout_unique_designs_used": r.get("holdout_unique_designs_used"),
                    "holdout_t1_mae_unique_design": r.get("holdout_t1_mae_unique_design"),
                    "holdout_t2_log10_mae_unique_design": r.get("holdout_t2_log10_mae_unique_design"),
                    "holdout_mode": r.get("holdout_mode"),
                }
            )

    claim_df = pd.DataFrame(claim_rows)
    claim_df.to_csv(paper_dir / "phase7_paper_claims_table.csv", index=False)

    lines: List[str] = []
    lines.append("# Phase 7 Evidence Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Top variant by multi-seed reliability: `{top_variant}`" if top_variant else "- Multi-seed summary unavailable")
    lines.append(f"- Reference Phase 6 summary: `{phase6_summary_path.resolve()}`")
    lines.append("- Claim scope: reliability-aware ranking, uncertainty gating, and shortlist quality under data shift.")
    lines.append("- Out of scope: absolute coherence prediction guarantees without substantial geometry-linked measured data.")
    lines.append("")

    if multiseed_summary is not None and not multiseed_summary.empty:
        lines.append("## Multi-Seed Stability")
        lines.append("")
        for _, r in multiseed_summary.iterrows():
            lines.append(
                "- "
                f"{r['variant']}: rank_mean={float(r.get('reliability_rank_mean', np.nan)):.3f}, "
                f"rank_std={float(r.get('reliability_rank_std', np.nan)):.3f}, "
                f"ood_t1_mae_mean={float(r.get('ood_t1_mae_mean', np.nan)):.6f}, "
                f"ood_t2_log10_mae_mean={float(r.get('ood_t2_log10_mae_mean', np.nan)):.6f}, "
                f"top1_count={int(r.get('top1_count', 0))}"
            )
        lines.append("")

    if holdout_summary is not None and not holdout_summary.empty:
        lines.append("## Source Holdout")
        lines.append("")
        h0 = holdout_summary.iloc[0]
        lines.append(
            "- "
            f"n_sources_total={int(h0.get('n_sources_total', 0))}, "
            f"n_sources_eligible={int(h0.get('n_sources_eligible', 0))}, "
            f"n_unique_designs_total={int(h0.get('n_unique_designs_total', 0))}, "
            f"n_unique_designs_eligible_sources={int(h0.get('n_unique_designs_eligible_sources', 0))}, "
            f"n_synthetic_rows_input={int(h0.get('n_synthetic_rows_input', 0))}"
        )
        for _, r in holdout_summary.iterrows():
            lines.append(
                "- "
                f"source={r['source_name']}, rows={int(r['source_rows'])}, "
                f"anchor_linked_excluded={int(r.get('train_rows_anchor_linked_excluded', 0))}, "
                f"holdout_t1_mae={float(r.get('holdout_t1_mae', np.nan)):.6f}, "
                f"holdout_t2_log10_mae={float(r.get('holdout_t2_log10_mae', np.nan)):.6f}, "
                f"unique_designs={int(r.get('holdout_unique_designs_used', 0))}, "
                f"leakage_overlap_count={int(r.get('leakage_overlap_count', 0))}, "
                f"holdout_t1_mae_unique_design={float(r.get('holdout_t1_mae_unique_design', np.nan)):.6f}, "
                f"holdout_t2_log10_mae_unique_design={float(r.get('holdout_t2_log10_mae_unique_design', np.nan)):.6f}"
            )
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append(f"- Multi-seed raw: `{outdir / 'phase7_multiseed_raw.csv'}`")
    lines.append(f"- Multi-seed summary: `{outdir / 'phase7_multiseed_summary.csv'}`")
    lines.append(f"- Holdout summary: `{outdir / 'source_holdout' / 'phase7_source_holdout_summary.csv'}`")
    lines.append(f"- Holdout overview: `{outdir / 'source_holdout' / 'phase7_source_holdout_overview.json'}`")
    lines.append(f"- Holdout leakage audit: `{outdir / 'source_holdout' / 'audits' / 'phase7_holdout_leakage_audit.csv'}`")
    lines.append(f"- Paper claims table: `{paper_dir / 'phase7_paper_claims_table.csv'}`")

    (outdir / "phase7_report.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_variant": top_variant,
        "phase6_summary": str(phase6_summary_path.resolve()),
        "multiseed_raw": None if multiseed_raw is None else str((outdir / "phase7_multiseed_raw.csv").resolve()),
        "multiseed_summary": None if multiseed_summary is None else str((outdir / "phase7_multiseed_summary.csv").resolve()),
        "holdout_summary": None if holdout_summary is None else str((outdir / "source_holdout" / "phase7_source_holdout_summary.csv").resolve()),
        "holdout_overview": str((outdir / "source_holdout" / "phase7_source_holdout_overview.json").resolve()),
        "holdout_leakage_audit": str((outdir / "source_holdout" / "audits" / "phase7_holdout_leakage_audit.csv").resolve()),
        "paper_claims_csv": str((paper_dir / "phase7_paper_claims_table.csv").resolve()),
    }
    (outdir / "phase7_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(args.seeds)

    multiseed_raw: Optional[pd.DataFrame] = None
    multiseed_summary: Optional[pd.DataFrame] = None
    holdout_summary: Optional[pd.DataFrame] = None

    if not args.skip_multiseed:
        multiseed_raw = run_multiseed_phase6(root=root, args=args, seeds=seeds, outdir=outdir)
        multiseed_summary = summarize_multiseed(multiseed_raw, outdir=outdir)

    if not args.skip_holdout:
        holdout_summary = run_source_holdout(root=root, args=args, outdir=outdir)

    if not args.skip_paper:
        build_paper_outputs(
            outdir=outdir,
            multiseed_raw=multiseed_raw,
            multiseed_summary=multiseed_summary,
            holdout_summary=holdout_summary,
            phase6_summary_path=args.phase6_summary,
        )

    print("=== Phase 7 Complete ===")
    if multiseed_summary is not None and not multiseed_summary.empty:
        print(f"multiseed_top_variant={multiseed_summary.iloc[0]['variant']}")
        print(f"multiseed_summary={outdir / 'phase7_multiseed_summary.csv'}")
    if holdout_summary is not None and not holdout_summary.empty:
        print(f"holdout_summary={outdir / 'source_holdout' / 'phase7_source_holdout_summary.csv'}")
    print(f"report={outdir / 'phase7_report.md'}")
    print(f"summary={outdir / 'phase7_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
