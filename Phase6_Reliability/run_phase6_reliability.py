
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

TARGET_COLS: Tuple[str, ...] = ("t1_us", "t2_us")


@dataclass
class VariantSpec:
    name: str
    measurement_csv: Path
    extra_train_args: List[str]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 6 reliability ablation + stress pipeline")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument(
        "--measurement-csv",
        type=Path,
        default=root / "Dataset" / "measurement_dataset_public_bootstrap.csv",
    )
    parser.add_argument("--phase2-bundle", type=Path, default=root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument(
        "--phase4-train-script",
        type=Path,
        default=root / "Phase4_Coherence" / "train_phase4_coherence.py",
    )
    parser.add_argument(
        "--phase4-validate-script",
        type=Path,
        default=root / "Phase4_Coherence" / "validate_phase4_coherence.py",
    )
    parser.add_argument(
        "--phase5-script",
        type=Path,
        default=root / "Phase5_ClosedLoop" / "run_phase5_closed_loop.py",
    )
    parser.add_argument("--phase1-model", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-meta", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")
    parser.add_argument("--phase5-targets-csv", type=Path, default=root / "Phase5_ClosedLoop" / "artifacts" / "phase5_targets_used.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))

    parser.add_argument("--high-trust-confidence-threshold", type=float, default=0.62)
    parser.add_argument("--phase5-num-targets", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-phase5", action="store_true")
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def find_first_present(columns: Sequence[str], options: Sequence[str]) -> Optional[str]:
    for c in options:
        if c in columns:
            return c
    return None


def load_measurement_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    if raw.empty:
        return None

    row_col = find_first_present(raw.columns, ["row_index"])
    design_col = find_first_present(raw.columns, ["design_id"])
    if row_col is None and design_col is None:
        return None

    t1_col = find_first_present(raw.columns, ["measured_t1_us", "t1_us", "T1_us", "T1"])
    t2_col = find_first_present(raw.columns, ["measured_t2_us", "t2_us", "T2_us", "T2"])
    if t1_col is None and t2_col is None:
        return None

    source_col = find_first_present(raw.columns, ["source_name", "source", "dataset_source"])

    keep_cols: List[str] = []
    if row_col is not None:
        keep_cols.append(row_col)
    if design_col is not None and design_col != row_col:
        keep_cols.append(design_col)
    if t1_col is not None:
        keep_cols.append(t1_col)
    if t2_col is not None and t2_col != t1_col:
        keep_cols.append(t2_col)
    if source_col is not None and source_col not in keep_cols:
        keep_cols.append(source_col)

    out = raw[keep_cols].copy()
    if row_col is not None and row_col != "row_index":
        out = out.rename(columns={row_col: "row_index"})
    if design_col is not None and design_col != "design_id":
        out = out.rename(columns={design_col: "design_id"})
    if t1_col is not None and t1_col != "measured_t1_us":
        out = out.rename(columns={t1_col: "measured_t1_us"})
    if t2_col is not None and t2_col != "measured_t2_us":
        out = out.rename(columns={t2_col: "measured_t2_us"})
    if source_col is not None and source_col != "source_name":
        out = out.rename(columns={source_col: "source_name"})

    if "row_index" in out.columns:
        out["row_index"] = pd.to_numeric(out["row_index"], errors="coerce")
    if "design_id" in out.columns:
        out["design_id"] = pd.to_numeric(out["design_id"], errors="coerce")
    if "measured_t1_us" in out.columns:
        out["measured_t1_us"] = pd.to_numeric(out["measured_t1_us"], errors="coerce")
    if "measured_t2_us" in out.columns:
        out["measured_t2_us"] = pd.to_numeric(out["measured_t2_us"], errors="coerce")
    if "source_name" in out.columns:
        out["source_name"] = out["source_name"].fillna("").astype(str)
    return out


def _apply_source_offsets(values: np.ndarray, sources: np.ndarray, target: str, source_calibration: Optional[Dict[str, object]]) -> np.ndarray:
    if not source_calibration:
        return values
    targets = source_calibration.get("targets", {})
    if not isinstance(targets, dict):
        return values
    tcal = targets.get(target, {})
    if not isinstance(tcal, dict):
        return values
    offsets = tcal.get("offsets_log10", {})
    if not isinstance(offsets, dict) or not offsets:
        return values

    out = values.copy()
    src = pd.Series(sources).fillna("").astype(str).to_numpy(dtype=object)
    for s, off in offsets.items():
        mask = np.isfinite(out) & (src == str(s))
        if int(mask.sum()) == 0:
            continue
        out[mask] = out[mask] / (10.0 ** float(off))
    return out


def derive_targets(
    base: pd.DataFrame,
    measurement_df: Optional[pd.DataFrame],
    mode: str,
    source_calibration: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    proxy_t1 = base["t1_estimate_us"].to_numpy(dtype=float)
    proxy_t2 = base["t2_estimate_us"].to_numpy(dtype=float)

    measured_t1 = np.full(len(base), np.nan, dtype=float)
    measured_t2 = np.full(len(base), np.nan, dtype=float)
    measured_source = np.array(["" for _ in range(len(base))], dtype=object)
    if measurement_df is not None and len(measurement_df) > 0:
        mdf = measurement_df.copy()
        if "row_index" in mdf.columns and np.isfinite(mdf["row_index"]).any():
            mdf = mdf.dropna(subset=["row_index"])
            mdf["row_index"] = mdf["row_index"].astype(int)
            mdf = mdf.drop_duplicates(subset=["row_index"], keep="last")
            m = mdf.set_index("row_index")
            measured_t1 = base["row_index"].map(m.get("measured_t1_us")).to_numpy(dtype=float)
            measured_t2 = base["row_index"].map(m.get("measured_t2_us")).to_numpy(dtype=float)
            if "source_name" in m.columns:
                measured_source = base["row_index"].map(m.get("source_name")).fillna("").astype(str).to_numpy(dtype=object)
        elif "design_id" in mdf.columns and np.isfinite(mdf["design_id"]).any():
            mdf = mdf.dropna(subset=["design_id"])
            mdf["design_id"] = mdf["design_id"].astype(int)
            mdf = mdf.drop_duplicates(subset=["design_id"], keep="last")
            m = mdf.set_index("design_id")
            measured_t1 = base["design_id"].map(m.get("measured_t1_us")).to_numpy(dtype=float)
            measured_t2 = base["design_id"].map(m.get("measured_t2_us")).to_numpy(dtype=float)
            if "source_name" in m.columns:
                measured_source = base["design_id"].map(m.get("source_name")).fillna("").astype(str).to_numpy(dtype=object)

    measured_t1 = _apply_source_offsets(measured_t1, measured_source, "t1_us", source_calibration)
    measured_t2 = _apply_source_offsets(measured_t2, measured_source, "t2_us", source_calibration)

    if mode == "proxy":
        t1 = proxy_t1
        t2 = proxy_t2
    elif mode == "hybrid":
        t1 = np.where(np.isfinite(measured_t1), measured_t1, proxy_t1)
        t2 = np.where(np.isfinite(measured_t2), measured_t2, proxy_t2)
    elif mode == "measured_only":
        t1 = measured_t1
        t2 = measured_t2
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    out = base.copy()
    out["t1_us"] = t1
    out["t2_us"] = t2
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    return out


def invert_target_transform(y_model: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    y = y_model.copy()
    if bool(transform["use_log"]):
        y = np.power(10.0, y)
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    return np.clip(y, lo, hi)


def predict_target_quantiles(models: Dict[str, object], target: str, x: np.ndarray, transform: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = invert_target_transform(models[f"{target}_q10"].predict(x), transform)
    q50 = invert_target_transform(models[f"{target}_q50"].predict(x), transform)
    q90 = invert_target_transform(models[f"{target}_q90"].predict(x), transform)
    stacked = np.vstack([q10, q50, q90]).T
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


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

def build_high_trust_measurement_csv(
    input_csv: Path,
    out_csv: Path,
    out_report: Path,
    confidence_threshold: float,
) -> Path:
    df = pd.read_csv(input_csv)
    if df.empty:
        raise RuntimeError(f"Input measurement CSV is empty: {input_csv}")

    match_method = df.get("match_method")
    source_name = df.get("source_name")
    confidence = pd.to_numeric(df.get("confidence_weight"), errors="coerce")

    c_match = match_method.fillna("").eq("freq+anh") if match_method is not None else pd.Series(False, index=df.index)
    c_squadds = source_name.fillna("").str.contains("SQuADDS", regex=False) if source_name is not None else pd.Series(False, index=df.index)
    c_trace = source_name.fillna("").str.contains("tracefit", case=False, regex=False) if source_name is not None else pd.Series(False, index=df.index)
    c_conf = confidence >= float(confidence_threshold)

    mask = c_match | c_squadds | c_trace | c_conf
    high = df.loc[mask].copy()

    if "row_index" in high.columns and high["row_index"].notna().any():
        high["row_index"] = pd.to_numeric(high["row_index"], errors="coerce")
        high = high.dropna(subset=["row_index"])
        high["row_index"] = high["row_index"].astype(int)
        high = high.drop_duplicates(subset=["row_index"], keep="first")

    if len(high) < 5:
        if confidence.notna().sum() > 0:
            high = (
                df.assign(_cw=confidence.fillna(-1.0))
                .sort_values("_cw", ascending=False)
                .head(max(5, min(25, len(df))))
                .drop(columns=["_cw"])
                .copy()
            )
        else:
            high = df.head(max(5, min(25, len(df)))).copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    high.to_csv(out_csv, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(out_csv.resolve()),
        "input_rows": int(len(df)),
        "output_rows": int(len(high)),
        "confidence_threshold": float(confidence_threshold),
        "kept_rules": {
            "match_method_eq_freq+anh": int(c_match.sum()),
            "source_contains_SQuADDS": int(c_squadds.sum()),
            "source_contains_tracefit": int(c_trace.sum()),
            "confidence_weight_ge_threshold": int(c_conf.sum()),
        },
        "source_counts": {
            str(k): int(v) for k, v in high.get("source_name", pd.Series(dtype=object)).value_counts(dropna=False).to_dict().items()
        },
    }
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_csv


def compute_calibration_gap(report: Mapping[str, object], split: str, target: str) -> float:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        return float("nan")
    sp = metrics.get(split, {})
    if not isinstance(sp, dict):
        return float("nan")
    tm = sp.get(target, {})
    if not isinstance(tm, dict):
        return float("nan")
    q10 = float(tm.get("calibration_p_le_q10", np.nan))
    q90 = float(tm.get("calibration_p_le_q90", np.nan))
    return float(abs(q10 - 0.10) + abs(q90 - 0.90))


def risk_coverage_curve(error: np.ndarray, uncertainty: np.ndarray, coverages: np.ndarray) -> Tuple[List[Dict[str, float]], float]:
    order = np.argsort(uncertainty)
    e_sorted = error[order]
    rows: List[Dict[str, float]] = []
    risks: List[float] = []
    for c in coverages:
        k = max(1, int(np.ceil(c * len(e_sorted))))
        risk = float(np.mean(e_sorted[:k]))
        risks.append(risk)
        rows.append({"coverage": float(c), "risk": risk, "n": int(k)})
    aurc = float(np.trapz(np.asarray(risks), x=coverages))
    return rows, aurc


def compute_bundle_risk_metrics(
    bundle_path: Path,
    single_csv: Path,
    measurement_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    bundle = joblib.load(bundle_path)
    feature_cols = list(bundle["feature_cols"])
    target_cols = list(bundle["target_cols"])
    target_transforms = dict(bundle["target_transforms"])
    row_index_all = list(bundle["row_index_all"])
    label_mode = str(bundle.get("label_mode_effective", "proxy"))

    df_raw = pd.read_csv(single_csv)
    ensure_required_columns(df_raw, ["design_id", *feature_cols, "t1_estimate_us", "t2_estimate_us"])
    base = df_raw[["design_id", *feature_cols, "t1_estimate_us", "t2_estimate_us"]].copy()
    base["row_index"] = df_raw.index.to_numpy()
    base = base.replace([np.inf, -np.inf], np.nan).dropna()

    measurement_df = load_measurement_df(measurement_csv)
    source_calibration = bundle.get("source_calibration", {})
    data = derive_targets(base, measurement_df, label_mode, source_calibration=source_calibration)
    frame = data.set_index("row_index")
    data = frame.loc[row_index_all].reset_index()

    x_raw = data.loc[:, feature_cols].to_numpy(dtype=float)
    y_raw = data.loc[:, target_cols].to_numpy(dtype=float)

    y_eval = np.zeros_like(y_raw)
    for i, t in enumerate(target_cols):
        lo = float(target_transforms[t]["clip_low"])
        hi = float(target_transforms[t]["clip_high"])
        y_eval[:, i] = np.clip(y_raw[:, i], lo, hi)

    x_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
    x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

    models = bundle["models"]
    q10_t1, q50_t1, q90_t1 = predict_target_quantiles(models, "t1_us", x_scaled, target_transforms["t1_us"])
    q10_t2, q50_t2, q90_t2 = predict_target_quantiles(models, "t2_us", x_scaled, target_transforms["t2_us"])

    width_t1 = q90_t1 - q10_t1
    width_t2 = q90_t2 - q10_t2
    unc = ((width_t1 - width_t1.mean()) / (width_t1.std() + 1e-12)) + ((width_t2 - width_t2.mean()) / (width_t2.std() + 1e-12))

    err_t1 = np.abs(y_eval[:, 0] - q50_t1)
    y2 = np.maximum(y_eval[:, 1], 1e-18)
    p2 = np.maximum(q50_t2, 1e-18)
    err_t2 = np.abs(np.log10(y2) - np.log10(p2))

    split_map = {
        "id_test": np.asarray(bundle["id_test_positions"], dtype=int),
        "ood": np.asarray(bundle["ood_positions"], dtype=int),
    }
    split_map["eval_all"] = np.unique(np.concatenate([split_map["id_test"], split_map["ood"]]))
    coverages = np.linspace(0.1, 1.0, 10)

    curve_rows: List[Dict[str, float]] = []
    summary: Dict[str, float] = {}
    for split_name, pos in split_map.items():
        for target, err in (("t1_us", err_t1), ("t2_us_log10", err_t2)):
            rows, aurc = risk_coverage_curve(err[pos], unc[pos], coverages)
            for r in rows:
                curve_rows.append(
                    {
                        "split": split_name,
                        "target": target,
                        "coverage": float(r["coverage"]),
                        "risk": float(r["risk"]),
                        "n": int(r["n"]),
                    }
                )
            r80 = next((x["risk"] for x in rows if abs(x["coverage"] - 0.8) < 1e-9), np.nan)
            r100 = next((x["risk"] for x in rows if abs(x["coverage"] - 1.0) < 1e-9), np.nan)
            summary[f"{split_name}_{target}_aurc"] = float(aurc)
            summary[f"{split_name}_{target}_risk_at_80"] = float(r80)
            summary[f"{split_name}_{target}_risk_at_100"] = float(r100)
            summary[f"{split_name}_{target}_abstain20_gain"] = float(r100 - r80)

    return pd.DataFrame(curve_rows), summary

def run_phase4_variant(
    root: Path,
    args: argparse.Namespace,
    variant: VariantSpec,
    phase4_runs_dir: Path,
) -> Dict[str, object]:
    outdir = phase4_runs_dir / variant.name
    outdir.mkdir(parents=True, exist_ok=True)
    log_train = outdir / "train.log"
    log_val = outdir / "validate.log"

    train_cmd = [
        str(args.python_bin),
        str(args.phase4_train_script),
        "--single-csv",
        str(args.single_csv),
        "--measurement-csv",
        str(variant.measurement_csv),
        "--phase2-bundle",
        str(args.phase2_bundle),
        "--output-dir",
        str(outdir),
        "--random-state",
        str(args.random_state),
        *variant.extra_train_args,
    ]
    run_cmd(train_cmd, workdir=root, log_path=log_train)

    bundle_path = outdir / "phase4_coherence_bundle.joblib"
    val_cmd = [
        str(args.python_bin),
        str(args.phase4_validate_script),
        "--single-csv",
        str(args.single_csv),
        "--measurement-csv",
        str(variant.measurement_csv),
        "--bundle-path",
        str(bundle_path),
        "--output-dir",
        str(outdir),
    ]
    run_cmd(val_cmd, workdir=root, log_path=log_val)

    training_summary_path = outdir / "phase4_training_summary.json"
    validation_path = outdir / "phase4_validation_report.json"
    tr = json.loads(training_summary_path.read_text(encoding="utf-8"))
    vr = json.loads(validation_path.read_text(encoding="utf-8"))

    curve_df, risk_summary = compute_bundle_risk_metrics(
        bundle_path=bundle_path,
        single_csv=args.single_csv,
        measurement_csv=variant.measurement_csv,
    )
    curve_df.insert(0, "variant", variant.name)
    curve_df.to_csv(outdir / "risk_coverage_curve.csv", index=False)

    metrics_ood = vr["metrics"]["ood"]
    metrics_id = vr["metrics"]["id_test"]
    ood_det = vr.get("ood_detection", {})
    combined_raw = ood_det.get("combined_or") if isinstance(ood_det, dict) else None
    combined = combined_raw if isinstance(combined_raw, dict) else {}

    rec: Dict[str, object] = {
        "variant": variant.name,
        "measurement_csv": str(variant.measurement_csv.resolve()),
        "rows_total": int(vr["rows"]["total"]),
        "rows_train": int(vr["rows"]["train"]),
        "rows_id_test": int(vr["rows"]["id_test"]),
        "rows_ood": int(vr["rows"]["ood"]),
        "label_mode_effective": tr.get("label_mode_effective"),
        "measurement_rows_matched": int(tr.get("label_stats", {}).get("measurement_rows_matched", 0)),
        "rows_with_measured_t1": int(tr.get("label_stats", {}).get("rows_with_measured_t1", 0)),
        "rows_with_measured_t2": int(tr.get("label_stats", {}).get("rows_with_measured_t2", 0)),
        "id_t1_mae": float(metrics_id["t1_us"]["mae"]),
        "ood_t1_mae": float(metrics_ood["t1_us"]["mae"]),
        "id_t2_log10_mae": float(metrics_id["t2_us"]["log10_mae"]),
        "ood_t2_log10_mae": float(metrics_ood["t2_us"]["log10_mae"]),
        "id_t1_cov80": float(metrics_id["t1_us"]["interval_80_coverage"]),
        "ood_t1_cov80": float(metrics_ood["t1_us"]["interval_80_coverage"]),
        "id_t2_cov80": float(metrics_id["t2_us"]["interval_80_coverage"]),
        "ood_t2_cov80": float(metrics_ood["t2_us"]["interval_80_coverage"]),
        "cal_gap_id_t1": compute_calibration_gap(vr, "id_test", "t1_us"),
        "cal_gap_ood_t1": compute_calibration_gap(vr, "ood", "t1_us"),
        "cal_gap_id_t2": compute_calibration_gap(vr, "id_test", "t2_us"),
        "cal_gap_ood_t2": compute_calibration_gap(vr, "ood", "t2_us"),
        "ood_f1_combined": float(combined.get("f1", np.nan)),
        "ood_precision_combined": float(combined.get("precision", np.nan)),
        "ood_recall_combined": float(combined.get("recall", np.nan)),
        "artifact_dir": str(outdir.resolve()),
    }
    for k, v in risk_summary.items():
        rec[k] = float(v)
    return rec


def add_reliability_rank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rank_cols_low = [
        "ood_t1_mae",
        "ood_t2_log10_mae",
        "cal_gap_ood_t1",
        "cal_gap_ood_t2",
        "eval_all_t1_us_aurc",
        "eval_all_t2_us_log10_aurc",
    ]
    rank_cols_high = ["ood_f1_combined", "eval_all_t1_us_abstain20_gain", "eval_all_t2_us_log10_abstain20_gain"]

    score = np.zeros(len(out), dtype=float)

    def _rank_with_nan(series: pd.Series, ascending: bool) -> np.ndarray:
        r = series.rank(method="min", ascending=ascending)
        fallback = float(r.max() + 1.0) if np.isfinite(r.max()) else float(len(series) + 1)
        r = r.fillna(fallback)
        return r.to_numpy(dtype=float)

    for c in rank_cols_low:
        if c in out.columns:
            score += _rank_with_nan(out[c], ascending=True)
    for c in rank_cols_high:
        if c in out.columns:
            score += _rank_with_nan(out[c], ascending=False)
    out["reliability_rank_sum"] = score
    out = out.sort_values(["reliability_rank_sum", "ood_t1_mae", "ood_t2_log10_mae"], ascending=[True, True, True]).reset_index(drop=True)
    out["reliability_rank"] = np.arange(1, len(out) + 1)
    return out


def run_phase5_experiment(
    root: Path,
    args: argparse.Namespace,
    run_name: str,
    phase4_bundle: Path,
    extra_args: Sequence[str],
    phase5_runs_dir: Path,
) -> Dict[str, object]:
    outdir = phase5_runs_dir / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"

    cmd = [
        str(args.python_bin),
        str(args.phase5_script),
        "--single-csv",
        str(args.single_csv),
        "--phase2-bundle",
        str(args.phase2_bundle),
        "--phase1-model",
        str(args.phase1_model),
        "--phase1-meta",
        str(args.phase1_meta),
        "--phase4-bundle",
        str(phase4_bundle),
        "--output-dir",
        str(outdir),
        "--num-targets",
        str(args.phase5_num_targets),
        "--random-state",
        str(args.random_state),
        *list(extra_args),
    ]
    if args.phase5_targets_csv.exists():
        cmd.extend(["--targets-csv", str(args.phase5_targets_csv)])

    run_cmd(cmd, workdir=root, log_path=log_path)

    summary = json.loads((outdir / "phase5_summary.json").read_text(encoding="utf-8"))
    cand = pd.read_csv(outdir / "phase5_candidate_batch.csv")
    sel = pd.read_csv(outdir / "phase5_selected_candidates.csv")

    pass_rate = float(cand["pass_all"].mean()) if len(cand) > 0 else float("nan")
    selected_uncertainty_rate = float((~sel["pass_uncertainty"]).mean()) if "pass_uncertainty" in sel.columns and len(sel) > 0 else float("nan")
    selected_low_conf_rate = float((sel["coh_confidence"] == "low").mean()) if "coh_confidence" in sel.columns and len(sel) > 0 else float("nan")

    return {
        "run_name": run_name,
        "phase4_bundle": str(phase4_bundle.resolve()),
        "allow_ood": bool(summary["config"]["allow_ood"]),
        "allow_low_confidence": bool(summary["config"]["allow_low_confidence"]),
        "allow_uncertain": bool(summary["config"]["allow_uncertain"]),
        "targets_total": int(summary["targets_total"]),
        "candidates_total": int(summary["candidates_total"]),
        "pass_all_count": int(summary["candidates_pass_all"]),
        "pass_all_rate": pass_rate,
        "selected_total": int(summary["selected_total"]),
        "selected_t1_p10_median": float(summary["metrics"]["selected_t1_p10_median"]),
        "selected_t1_p10_min": float(summary["metrics"]["selected_t1_p10_min"]),
        "selected_objective_median": float(summary["metrics"]["selected_objective_median"]),
        "selected_ood_rate": float(summary["metrics"]["selected_ood_rate"]),
        "selected_low_conf_rate": selected_low_conf_rate,
        "selected_uncertainty_fail_rate": selected_uncertainty_rate,
        "artifact_dir": str(outdir.resolve()),
    }

def build_variants(args: argparse.Namespace, high_trust_csv: Path) -> List[VariantSpec]:
    return [
        VariantSpec(
            name="proxy_baseline",
            measurement_csv=args.measurement_csv,
            extra_train_args=[
                "--label-mode",
                "proxy",
            ],
        ),
        VariantSpec(
            name="hybrid_full_conf",
            measurement_csv=args.measurement_csv,
            extra_train_args=[
                "--label-mode",
                "hybrid",
                "--measured-weight",
                "3.0",
                "--proxy-weight",
                "1.0",
            ],
        ),
        VariantSpec(
            name="hybrid_no_conf_weight",
            measurement_csv=args.measurement_csv,
            extra_train_args=[
                "--label-mode",
                "hybrid",
                "--measured-weight",
                "3.0",
                "--proxy-weight",
                "1.0",
                "--measurement-weight-min",
                "1.0",
                "--measurement-weight-max",
                "1.0",
            ],
        ),
        VariantSpec(
            name="hybrid_no_embedding_ood",
            measurement_csv=args.measurement_csv,
            extra_train_args=[
                "--label-mode",
                "hybrid",
                "--measured-weight",
                "3.0",
                "--proxy-weight",
                "1.0",
                "--disable-embedding-ood",
            ],
        ),
        VariantSpec(
            name="hybrid_high_trust",
            measurement_csv=high_trust_csv,
            extra_train_args=[
                "--label-mode",
                "hybrid",
                "--measured-weight",
                "3.0",
                "--proxy-weight",
                "1.0",
            ],
        ),
    ]


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    phase4_summary: pd.DataFrame,
    curve_df: pd.DataFrame,
    phase5_summary: Optional[pd.DataFrame],
    primary_variant: str,
    secondary_variant: Optional[str],
) -> None:
    report_lines: List[str] = []
    report_lines.append("# Phase 6 Reliability Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("")
    report_lines.append("## Scope")
    report_lines.append("")
    report_lines.append("- Purpose: reliability validation for publishable and lab-usable confidence claims.")
    report_lines.append(f"- Phase 4 variants evaluated: {len(phase4_summary)}")
    report_lines.append(f"- Phase 5 stress runs evaluated: {0 if phase5_summary is None else len(phase5_summary)}")
    report_lines.append("")
    report_lines.append("## Phase 4 Ablation Ranking")
    report_lines.append("")
    for _, r in phase4_summary.iterrows():
        report_lines.append(
            "- "
            f"{r['variant']}: rank={int(r['reliability_rank'])}, "
            f"ood_t1_mae={float(r['ood_t1_mae']):.6f}, "
            f"ood_t2_log10_mae={float(r['ood_t2_log10_mae']):.6f}, "
            f"ood_f1={float(r['ood_f1_combined']):.4f}, "
            f"cal_gap_t1={float(r['cal_gap_ood_t1']):.4f}, "
            f"cal_gap_t2={float(r['cal_gap_ood_t2']):.4f}"
        )
    report_lines.append("")
    report_lines.append("## Selected Models")
    report_lines.append("")
    report_lines.append(f"- Primary Phase 4 bundle: `{primary_variant}`")
    if secondary_variant is not None:
        report_lines.append(f"- Secondary comparator: `{secondary_variant}`")
    report_lines.append("")
    report_lines.append("## Risk-Coverage Highlights")
    report_lines.append("")

    best_curve = curve_df[curve_df["variant"] == primary_variant].copy()
    if not best_curve.empty:
        for target in ["t1_us", "t2_us_log10"]:
            subset = best_curve[(best_curve["split"] == "eval_all") & (best_curve["target"] == target)]
            if subset.empty:
                continue
            r80 = float(subset.loc[np.isclose(subset["coverage"], 0.8), "risk"].iloc[0])
            r100 = float(subset.loc[np.isclose(subset["coverage"], 1.0), "risk"].iloc[0])
            report_lines.append(f"- {target}: risk@80%={r80:.6f}, risk@100%={r100:.6f}, abstain20 gain={r100-r80:.6f}")
    report_lines.append("")

    if phase5_summary is not None and not phase5_summary.empty:
        report_lines.append("## Phase 5 Stress Results")
        report_lines.append("")
        for _, r in phase5_summary.iterrows():
            report_lines.append(
                "- "
                f"{r['run_name']}: pass_rate={float(r['pass_all_rate']):.4f}, "
                f"selected_t1_p10_median={float(r['selected_t1_p10_median']):.6f}, "
                f"selected_ood_rate={float(r['selected_ood_rate']):.4f}, "
                f"selected_low_conf_rate={float(r['selected_low_conf_rate']):.4f}"
            )
        report_lines.append("")

    report_lines.append("## Paper-Ready Artifacts")
    report_lines.append("")
    report_lines.append(f"- Variant summary CSV: `{outdir / 'phase6_phase4_variant_summary.csv'}`")
    report_lines.append(f"- Risk-coverage CSV: `{outdir / 'phase6_risk_coverage_curves.csv'}`")
    if phase5_summary is not None:
        report_lines.append(f"- Phase 5 stress CSV: `{outdir / 'phase6_phase5_stress_summary.csv'}`")
    report_lines.append(f"- Machine-readable summary: `{outdir / 'phase6_summary.json'}`")
    report_lines.append("")
    report_lines.append("## Limits")
    report_lines.append("")
    report_lines.append("- Reliability is still bounded by sparse true measured geometry-linked coherence labels.")
    report_lines.append("- Reported gains should be framed as risk-ranking gains, not final absolute-coherence guarantees.")

    (outdir / "phase6_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "single_csv": str(args.single_csv.resolve()),
            "measurement_csv": str(args.measurement_csv.resolve()),
            "phase5_targets_csv": str(args.phase5_targets_csv.resolve()) if args.phase5_targets_csv.exists() else None,
        },
        "primary_variant": primary_variant,
        "secondary_variant": secondary_variant,
        "phase4_variants": phase4_summary.to_dict(orient="records"),
        "phase5_stress_runs": None if phase5_summary is None else phase5_summary.to_dict(orient="records"),
    }
    (outdir / "phase6_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    phase4_runs_dir = outdir / "phase4_runs"
    phase5_runs_dir = outdir / "phase5_runs"
    datasets_dir = outdir / "datasets"
    phase4_runs_dir.mkdir(parents=True, exist_ok=True)
    phase5_runs_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    high_trust_csv = datasets_dir / "measurement_dataset_high_trust.csv"
    high_trust_report = datasets_dir / "measurement_dataset_high_trust.report.json"
    build_high_trust_measurement_csv(
        input_csv=args.measurement_csv,
        out_csv=high_trust_csv,
        out_report=high_trust_report,
        confidence_threshold=args.high_trust_confidence_threshold,
    )

    variants = build_variants(args, high_trust_csv)

    variant_rows: List[Dict[str, object]] = []
    curve_rows: List[pd.DataFrame] = []
    for v in variants:
        rec = run_phase4_variant(root=root, args=args, variant=v, phase4_runs_dir=phase4_runs_dir)
        variant_rows.append(rec)
        cpath = Path(rec["artifact_dir"]) / "risk_coverage_curve.csv"
        if cpath.exists():
            curve_rows.append(pd.read_csv(cpath))

    phase4_df = pd.DataFrame(variant_rows)
    phase4_df = add_reliability_rank(phase4_df)
    phase4_df.to_csv(outdir / "phase6_phase4_variant_summary.csv", index=False)

    if curve_rows:
        curves_df = pd.concat(curve_rows, ignore_index=True)
    else:
        curves_df = pd.DataFrame(columns=["variant", "split", "target", "coverage", "risk", "n"])
    curves_df.to_csv(outdir / "phase6_risk_coverage_curves.csv", index=False)

    primary_variant = str(phase4_df.iloc[0]["variant"])
    secondary_variant = str(phase4_df.iloc[1]["variant"]) if len(phase4_df) > 1 else None

    phase5_df: Optional[pd.DataFrame] = None
    if not args.skip_phase5:
        bundle_map = {
            str(r["variant"]): Path(str(r["artifact_dir"])) / "phase4_coherence_bundle.joblib"
            for r in phase4_df.to_dict(orient="records")
        }
        phase5_specs: List[Tuple[str, Path, List[str]]] = [
            (f"{primary_variant}__strict", bundle_map[primary_variant], []),
            (f"{primary_variant}__allow_ood", bundle_map[primary_variant], ["--allow-ood"]),
            (f"{primary_variant}__allow_low_conf", bundle_map[primary_variant], ["--allow-low-confidence"]),
            (f"{primary_variant}__allow_uncertain", bundle_map[primary_variant], ["--allow-uncertain"]),
            (
                f"{primary_variant}__relaxed_all",
                bundle_map[primary_variant],
                ["--allow-ood", "--allow-low-confidence", "--allow-uncertain"],
            ),
        ]
        if secondary_variant is not None:
            phase5_specs.insert(1, (f"{secondary_variant}__strict", bundle_map[secondary_variant], []))

        phase5_rows: List[Dict[str, object]] = []
        for run_name, bpath, extra in phase5_specs:
            rec = run_phase5_experiment(
                root=root,
                args=args,
                run_name=run_name,
                phase4_bundle=bpath,
                extra_args=extra,
                phase5_runs_dir=phase5_runs_dir,
            )
            phase5_rows.append(rec)

        phase5_df = pd.DataFrame(phase5_rows)
        phase5_df.to_csv(outdir / "phase6_phase5_stress_summary.csv", index=False)

    write_report(
        outdir=outdir,
        args=args,
        phase4_summary=phase4_df,
        curve_df=curves_df,
        phase5_summary=phase5_df,
        primary_variant=primary_variant,
        secondary_variant=secondary_variant,
    )

    print("=== Phase 6 Reliability Complete ===")
    print(f"primary_variant={primary_variant}")
    print(f"phase4_summary={outdir / 'phase6_phase4_variant_summary.csv'}")
    print(f"risk_coverage={outdir / 'phase6_risk_coverage_curves.csv'}")
    if phase5_df is not None:
        print(f"phase5_stress={outdir / 'phase6_phase5_stress_summary.csv'}")
    print(f"report={outdir / 'phase6_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



