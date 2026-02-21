#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

GEOMETRY_COLS: Tuple[str, ...] = (
    "pad_width_um",
    "pad_height_um",
    "gap_um",
    "junction_area_um2",
)

PHYSICS_COLS: Tuple[str, ...] = (
    "freq_01_GHz",
    "anharmonicity_GHz",
    "EJ_GHz",
    "EC_GHz",
    "charge_sensitivity_GHz",
    "EJ_EC_ratio",
)

FEATURE_COLS: Tuple[str, ...] = GEOMETRY_COLS + PHYSICS_COLS
TARGET_COLS: Tuple[str, ...] = ("t1_us", "t2_us")
QUANTILES: Tuple[float, ...] = (0.10, 0.50, 0.90)


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


def encode_geom_in_batches(model: TwinEncoder, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    out: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(dtype=torch.float32)
            zb = model.encode_geom(xb).cpu().numpy()
            out.append(zb)
    return l2_normalize(np.vstack(out))


def ensure_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def quantile_metrics(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y - q50)
    width = q90 - q10
    corr = spearmanr(abs_err, width, nan_policy="omit").correlation

    y_safe = np.maximum(y, 1e-18)
    p_safe = np.maximum(q50, 1e-18)
    log_mae = float(np.mean(np.abs(np.log10(y_safe) - np.log10(p_safe))))

    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((y - q50) ** 2))),
        "log10_mae": log_mae,
        "pinball_q10": pinball_loss(y, q10, 0.10),
        "pinball_q50": pinball_loss(y, q50, 0.50),
        "pinball_q90": pinball_loss(y, q90, 0.90),
        "interval_80_coverage": float(np.mean((y >= q10) & (y <= q90))),
        "interval_width_mean": float(np.mean(width)),
        "interval_width_p90": float(np.quantile(width, 0.90)),
        "calibration_p_le_q10": float(np.mean(y <= q10)),
        "calibration_p_le_q90": float(np.mean(y <= q90)),
        "uncertainty_error_spearman": float(0.0 if np.isnan(corr) else corr),
    }


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

    weight_col = find_first_present(raw.columns, ["confidence_weight", "sample_weight", "weight"])
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
    if weight_col is not None and weight_col not in keep_cols:
        keep_cols.append(weight_col)
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
    if weight_col is not None and weight_col != "confidence_weight":
        out = out.rename(columns={weight_col: "confidence_weight"})
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
    if "confidence_weight" in out.columns:
        out["confidence_weight"] = pd.to_numeric(out["confidence_weight"], errors="coerce")
        out.loc[out["confidence_weight"] <= 0, "confidence_weight"] = np.nan
    if "source_name" in out.columns:
        out["source_name"] = out["source_name"].fillna("").astype(str)

    return out


def _canonicalize_measured_by_source(
    values: np.ndarray,
    sources: np.ndarray,
    min_rows: int,
    max_abs_shift_log10: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = values.copy()
    values_f = np.asarray(values, dtype=float)
    sources_s = pd.Series(sources).fillna("").astype(str).to_numpy(dtype=object)

    valid = np.isfinite(values_f) & (values_f > 0)
    if int(valid.sum()) == 0:
        return out, {
            "rows_used": 0,
            "global_median_log10": None,
            "offsets_log10": {},
            "source_counts": {},
            "offset_std_log10": 0.0,
        }

    logs_all = np.log10(np.maximum(values_f[valid], 1e-18))
    global_median = float(np.median(logs_all))

    offsets: Dict[str, float] = {}
    source_counts: Dict[str, int] = {}

    unique_sources = sorted({str(s) for s in sources_s if str(s)})
    for src in unique_sources:
        mask = valid & (sources_s == src)
        n = int(mask.sum())
        source_counts[src] = n
        if n < int(min_rows):
            continue
        src_med = float(np.median(np.log10(np.maximum(values_f[mask], 1e-18))))
        off = float(np.clip(src_med - global_median, -abs(max_abs_shift_log10), abs(max_abs_shift_log10)))
        offsets[src] = off
        out[mask] = values_f[mask] / (10.0 ** off)

    offset_std = float(np.std(list(offsets.values()))) if offsets else 0.0
    return out, {
        "rows_used": int(valid.sum()),
        "global_median_log10": global_median,
        "offsets_log10": offsets,
        "source_counts": source_counts,
        "offset_std_log10": offset_std,
    }


def derive_targets(
    base: pd.DataFrame,
    measurement_df: Optional[pd.DataFrame],
    label_mode: str,
    source_calibration_min_rows: int,
    source_calibration_max_shift_log10: float,
) -> Tuple[pd.DataFrame, Dict[str, object], str, Dict[str, object]]:
    proxy_t1 = base["t1_estimate_us"].to_numpy(dtype=float)
    proxy_t2 = base["t2_estimate_us"].to_numpy(dtype=float)

    measured_t1 = np.full(len(base), np.nan, dtype=float)
    measured_t2 = np.full(len(base), np.nan, dtype=float)
    measured_w = np.full(len(base), np.nan, dtype=float)
    measured_source = np.array(["" for _ in range(len(base))], dtype=object)

    measured_rows = 0
    if measurement_df is not None and len(measurement_df) > 0:
        mdf = measurement_df.copy()
        id_col: Optional[str] = None

        if "row_index" in mdf.columns and np.isfinite(mdf["row_index"]).any():
            id_col = "row_index"
            mdf = mdf.dropna(subset=["row_index"])
            mdf["row_index"] = mdf["row_index"].astype(int)
            mdf = mdf.drop_duplicates(subset=["row_index"], keep="last")
            m = mdf.set_index("row_index")
            measured_t1 = base["row_index"].map(m.get("measured_t1_us")).to_numpy(dtype=float)
            measured_t2 = base["row_index"].map(m.get("measured_t2_us")).to_numpy(dtype=float)
            if "confidence_weight" in m.columns:
                measured_w = base["row_index"].map(m.get("confidence_weight")).to_numpy(dtype=float)
            if "source_name" in m.columns:
                measured_source = base["row_index"].map(m.get("source_name")).fillna("").astype(str).to_numpy(dtype=object)
        elif "design_id" in mdf.columns and np.isfinite(mdf["design_id"]).any():
            id_col = "design_id"
            mdf = mdf.dropna(subset=["design_id"])
            mdf["design_id"] = mdf["design_id"].astype(int)
            mdf = mdf.drop_duplicates(subset=["design_id"], keep="last")
            m = mdf.set_index("design_id")
            measured_t1 = base["design_id"].map(m.get("measured_t1_us")).to_numpy(dtype=float)
            measured_t2 = base["design_id"].map(m.get("measured_t2_us")).to_numpy(dtype=float)
            if "confidence_weight" in m.columns:
                measured_w = base["design_id"].map(m.get("confidence_weight")).to_numpy(dtype=float)
            if "source_name" in m.columns:
                measured_source = base["design_id"].map(m.get("source_name")).fillna("").astype(str).to_numpy(dtype=object)

        if id_col is not None:
            measured_rows = int(np.sum(np.isfinite(measured_t1) | np.isfinite(measured_t2)))

    measured_t1_canon, t1_cal = _canonicalize_measured_by_source(
        values=measured_t1,
        sources=measured_source,
        min_rows=source_calibration_min_rows,
        max_abs_shift_log10=source_calibration_max_shift_log10,
    )
    measured_t2_canon, t2_cal = _canonicalize_measured_by_source(
        values=measured_t2,
        sources=measured_source,
        min_rows=source_calibration_min_rows,
        max_abs_shift_log10=source_calibration_max_shift_log10,
    )

    source_calibration = {
        "enabled": True,
        "config": {
            "min_rows": int(source_calibration_min_rows),
            "max_abs_shift_log10": float(source_calibration_max_shift_log10),
        },
        "targets": {
            "t1_us": t1_cal,
            "t2_us": t2_cal,
        },
    }

    mode = label_mode
    if mode == "auto":
        mode = "hybrid" if measured_rows >= 30 else "proxy"

    if mode == "proxy":
        t1 = proxy_t1.copy()
        t2 = proxy_t2.copy()
    elif mode == "hybrid":
        t1 = np.where(np.isfinite(measured_t1_canon), measured_t1_canon, proxy_t1)
        t2 = np.where(np.isfinite(measured_t2_canon), measured_t2_canon, proxy_t2)
    elif mode == "measured_only":
        t1 = measured_t1_canon.copy()
        t2 = measured_t2_canon.copy()
    else:
        raise ValueError(f"Unsupported label mode: {mode}")

    out = base.copy()
    out["t1_us"] = t1
    out["t2_us"] = t2
    out["has_measured_t1"] = np.isfinite(measured_t1)
    out["has_measured_t2"] = np.isfinite(measured_t2)
    out["measurement_confidence_weight"] = np.where(np.isfinite(measured_w), measured_w, 1.0)
    out["measurement_confidence_weight"] = np.clip(out["measurement_confidence_weight"], 1e-3, 1e3)
    out["measurement_source_name"] = pd.Series(measured_source).fillna("").astype(str).to_numpy(dtype=object)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[*FEATURE_COLS, "t1_us", "t2_us"]).reset_index(drop=True)

    measured_conf = out.loc[(out["has_measured_t1"] | out["has_measured_t2"]), "measurement_confidence_weight"]

    stats = {
        "mode_requested": label_mode,
        "mode_effective": mode,
        "measurement_rows_matched": measured_rows,
        "rows_final": int(len(out)),
        "rows_with_measured_t1": int(np.sum(np.isfinite(measured_t1))),
        "rows_with_measured_t2": int(np.sum(np.isfinite(measured_t2))),
        "measurement_confidence_weight": {
            "count": int(measured_conf.shape[0]),
            "min": float(measured_conf.min()) if measured_conf.shape[0] > 0 else None,
            "median": float(measured_conf.median()) if measured_conf.shape[0] > 0 else None,
            "max": float(measured_conf.max()) if measured_conf.shape[0] > 0 else None,
        },
        "measurement_source_counts": {
            str(k): int(v)
            for k, v in out.loc[(out["has_measured_t1"] | out["has_measured_t2"]), "measurement_source_name"].value_counts().to_dict().items()
            if str(k)
        },
        "source_calibration": source_calibration,
    }
    return out, stats, mode, source_calibration


def build_ood_mask(df: pd.DataFrame, width_q: float, height_q: float, gap_q: float) -> Tuple[np.ndarray, Dict[str, float]]:
    width_thr = float(df["pad_width_um"].quantile(width_q))
    height_thr = float(df["pad_height_um"].quantile(height_q))
    gap_thr = float(df["gap_um"].quantile(gap_q))
    mask = (
        (df["pad_width_um"] >= width_thr)
        | (df["pad_height_um"] >= height_thr)
        | (df["gap_um"] <= gap_thr)
    ).to_numpy()
    thresholds = {
        "pad_width_um_q": width_q,
        "pad_width_um_threshold": width_thr,
        "pad_height_um_q": height_q,
        "pad_height_um_threshold": height_thr,
        "gap_um_q": gap_q,
        "gap_um_threshold": gap_thr,
    }
    return mask, thresholds


def apply_target_transform(y: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    clipped = np.clip(y, lo, hi)
    if bool(transform["use_log"]):
        clipped = np.log10(np.maximum(clipped, 1e-18))
    return clipped


def invert_target_transform(y_model: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    y = y_model.copy()
    if bool(transform["use_log"]):
        y = np.power(10.0, y)
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    return np.clip(y, lo, hi)


def predict_target_quantiles(
    models: Dict[str, GradientBoostingRegressor],
    target: str,
    x: np.ndarray,
    transform: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10_m = models[f"{target}_q10"].predict(x)
    q50_m = models[f"{target}_q50"].predict(x)
    q90_m = models[f"{target}_q90"].predict(x)

    q10 = invert_target_transform(q10_m, transform)
    q50 = invert_target_transform(q50_m, transform)
    q90 = invert_target_transform(q90_m, transform)

    stacked = np.vstack([q10, q50, q90]).T
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Phase 4 coherence predictor")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--measurement-csv", type=Path, default=root / "Dataset" / "measurement_dataset.csv")
    parser.add_argument("--phase2-bundle", type=Path, default=root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    parser.add_argument("--label-mode", choices=("auto", "proxy", "hybrid", "measured_only"), default="auto")
    parser.add_argument("--proxy-weight", type=float, default=1.0)
    parser.add_argument("--measured-weight", type=float, default=4.0)
    parser.add_argument("--measurement-weight-min", type=float, default=0.25)
    parser.add_argument("--measurement-weight-max", type=float, default=3.0)
    parser.add_argument("--source-calibration-min-rows", type=int, default=5)
    parser.add_argument("--source-calibration-max-shift-log10", type=float, default=1.0)

    parser.add_argument("--id-test-size", type=float, default=0.2)
    parser.add_argument("--ood-width-quantile", type=float, default=0.95)
    parser.add_argument("--ood-height-quantile", type=float, default=0.95)
    parser.add_argument("--ood-gap-quantile", type=float, default=0.05)

    parser.add_argument("--target-clip-low", type=float, default=0.001)
    parser.add_argument("--target-clip-high", type=float, default=0.995)
    parser.add_argument("--log-transform-ratio", type=float, default=1e4)

    parser.add_argument("--n-estimators", type=int, default=450)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=25)
    parser.add_argument("--subsample", type=float, default=0.9)

    parser.add_argument("--feature-ood-quantile", type=float, default=0.95)
    parser.add_argument("--embedding-ood-quantile", type=float, default=0.95)
    parser.add_argument("--disable-embedding-ood", action="store_true")

    parser.add_argument("--uncertain-top-n", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.random_state)

    df_raw = pd.read_csv(args.single_csv)
    ensure_required_columns(df_raw, ["design_id", *FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"])

    base = df_raw[["design_id", *FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"]].copy()
    base["row_index"] = df_raw.index.to_numpy()
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"]).reset_index(drop=True)

    measurement_df = load_measurement_df(args.measurement_csv)
    data, label_stats, mode_effective, source_calibration = derive_targets(
        base,
        measurement_df,
        args.label_mode,
        source_calibration_min_rows=args.source_calibration_min_rows,
        source_calibration_max_shift_log10=args.source_calibration_max_shift_log10,
    )
    if len(data) < 1000:
        raise SystemExit("Not enough rows for reliable Phase 4 training")

    ood_mask, ood_thresholds = build_ood_mask(
        data,
        width_q=args.ood_width_quantile,
        height_q=args.ood_height_quantile,
        gap_q=args.ood_gap_quantile,
    )

    all_pos = np.arange(len(data), dtype=int)
    in_pos = all_pos[~ood_mask]
    ood_pos = all_pos[ood_mask]
    if len(in_pos) < 200 or len(ood_pos) < 100:
        raise SystemExit("Split too small; adjust OOD quantiles")

    train_pos, id_test_pos = train_test_split(
        in_pos,
        test_size=args.id_test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    x_raw = data.loc[:, FEATURE_COLS].to_numpy(dtype=float)
    y_raw = data.loc[:, TARGET_COLS].to_numpy(dtype=float)

    scaler = StandardScaler().fit(x_raw[train_pos])
    x_mean = scaler.mean_.astype(np.float32)
    x_scale = np.where(scaler.scale_ <= 1e-12, 1.0, scaler.scale_).astype(np.float32)
    x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

    target_transforms: Dict[str, Dict[str, float]] = {}
    y_model_all = np.zeros_like(y_raw, dtype=float)
    y_eval_all = np.zeros_like(y_raw, dtype=float)

    for t_idx, target in enumerate(TARGET_COLS):
        train_target = y_raw[train_pos, t_idx]
        lo = float(np.quantile(train_target, args.target_clip_low))
        hi = float(np.quantile(train_target, args.target_clip_high))
        lo = max(lo, 1e-18)
        hi = max(hi, lo * (1.0 + 1e-9))
        ratio = hi / lo
        use_log = bool(ratio >= args.log_transform_ratio)

        transform = {
            "clip_low": lo,
            "clip_high": hi,
            "use_log": use_log,
            "ratio": ratio,
        }
        target_transforms[target] = transform

        y_eval = np.clip(y_raw[:, t_idx], lo, hi)
        y_eval_all[:, t_idx] = y_eval
        y_model_all[:, t_idx] = apply_target_transform(y_raw[:, t_idx], transform)

    models: Dict[str, GradientBoostingRegressor] = {}
    target_sample_weight_stats: Dict[str, Dict[str, float]] = {}
    for t_idx, target in enumerate(TARGET_COLS):
        y_train = y_model_all[train_pos, t_idx]

        if target == "t1_us":
            has_measured = data["has_measured_t1"].to_numpy(dtype=bool)
        else:
            has_measured = data["has_measured_t2"].to_numpy(dtype=bool)

        conf = data["measurement_confidence_weight"].to_numpy(dtype=float)
        conf = np.clip(conf, args.measurement_weight_min, args.measurement_weight_max)
        sample_weight_all = np.where(has_measured, args.measured_weight * conf, args.proxy_weight).astype(float)
        sample_weight_all = np.maximum(sample_weight_all, 1e-6)
        sample_weight_train = sample_weight_all[train_pos]

        target_sample_weight_stats[target] = {
            "train_min": float(np.min(sample_weight_train)),
            "train_median": float(np.median(sample_weight_train)),
            "train_max": float(np.max(sample_weight_train)),
            "train_mean": float(np.mean(sample_weight_train)),
            "train_fraction_measured": float(np.mean(has_measured[train_pos])),
        }

        for q in QUANTILES:
            key = f"{target}_q{int(q * 100)}"
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                subsample=args.subsample,
                random_state=args.random_state + 97 * t_idx + int(q * 1000),
            )
            model.fit(x_scaled[train_pos], y_train, sample_weight=sample_weight_train)
            models[key] = model

    def eval_split(positions: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        x = x_scaled[positions]
        for t_idx, target in enumerate(TARGET_COLS):
            y = y_eval_all[positions, t_idx]
            q10, q50, q90 = predict_target_quantiles(models, target, x, target_transforms[target])
            out[target] = quantile_metrics(y, q10, q50, q90)
        return out

    metrics_train = eval_split(train_pos)
    metrics_id = eval_split(id_test_pos)
    metrics_ood = eval_split(ood_pos)

    split = np.array(["train"] * len(data), dtype=object)
    split[id_test_pos] = "id_test"
    split[ood_pos] = "ood"

    nbr_feat = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nbr_feat.fit(x_scaled[train_pos])
    d_train, _ = nbr_feat.kneighbors(x_scaled[train_pos], n_neighbors=2)
    feat_train_nn = d_train[:, 1]
    feature_ood_threshold = float(np.quantile(feat_train_nn, args.feature_ood_quantile))

    embedding_ref: Dict[str, object] = {"enabled": False}
    if (not args.disable_embedding_ood) and args.phase2_bundle.exists():
        phase2_bundle = torch.load(args.phase2_bundle, map_location="cpu", weights_only=False)
        phase2_geom_cols = list(phase2_bundle["geometry_cols"])
        if all(c in data.columns for c in phase2_geom_cols):
            cfg = phase2_bundle["model_config"]
            twin = TwinEncoder(
                geom_in=int(cfg["geom_in"]),
                phys_in=int(cfg["phys_in"]),
                hidden_dim=int(cfg["hidden_dim"]),
                emb_dim=int(cfg["emb_dim"]),
                dropout=float(cfg["dropout"]),
            )
            twin.load_state_dict(phase2_bundle["state_dict"])
            twin.eval()

            g_mean = np.asarray(phase2_bundle["geom_scaler_mean"], dtype=np.float32)
            g_scale = np.asarray(phase2_bundle["geom_scaler_scale"], dtype=np.float32)

            xg = data.loc[:, phase2_geom_cols].to_numpy(dtype=float)
            xg_scaled = ((xg - g_mean) / g_scale).astype(np.float32)
            z_all = encode_geom_in_batches(twin, xg_scaled, batch_size=2048)
            z_train = z_all[train_pos]

            nbr_emb = NearestNeighbors(n_neighbors=2, metric="euclidean")
            nbr_emb.fit(z_train)
            dz, _ = nbr_emb.kneighbors(z_train, n_neighbors=2)
            emb_train_nn = dz[:, 1]
            emb_threshold = float(np.quantile(emb_train_nn, args.embedding_ood_quantile))

            embedding_ref = {
                "enabled": True,
                "geometry_cols": phase2_geom_cols,
                "model_config": copy.deepcopy(cfg),
                "state_dict": copy.deepcopy(phase2_bundle["state_dict"]),
                "geom_scaler_mean": g_mean,
                "geom_scaler_scale": g_scale,
                "train_embeddings": z_train.astype(np.float32),
                "ood_threshold": emb_threshold,
                "ood_quantile": float(args.embedding_ood_quantile),
            }

    feature_rows: List[Dict[str, float]] = []
    for col in FEATURE_COLS:
        f = data[col].to_numpy(dtype=float)
        corr_t1 = spearmanr(f, y_eval_all[:, 0], nan_policy="omit").correlation
        corr_t2 = spearmanr(f, y_eval_all[:, 1], nan_policy="omit").correlation
        corr_t1 = 0.0 if np.isnan(corr_t1) else float(corr_t1)
        corr_t2 = 0.0 if np.isnan(corr_t2) else float(corr_t2)

        t1_thr = float(np.quantile(y_eval_all[:, 0], 0.20))
        t2_thr = float(np.quantile(y_eval_all[:, 1], 0.20))
        low_t1 = y_eval_all[:, 0] <= t1_thr
        low_t2 = y_eval_all[:, 1] <= t2_thr

        std = float(np.std(f))
        std = 1.0 if std <= 1e-12 else std
        delta_t1 = float((np.mean(f[low_t1]) - np.mean(f[~low_t1])) / std)
        delta_t2 = float((np.mean(f[low_t2]) - np.mean(f[~low_t2])) / std)

        risk_score = float((abs(corr_t1) + abs(corr_t2) + abs(delta_t1) + abs(delta_t2)) / 4.0)
        feature_rows.append(
            {
                "feature": col,
                "spearman_t1": corr_t1,
                "spearman_t2": corr_t2,
                "low_t1_shift_std": delta_t1,
                "low_t2_shift_std": delta_t2,
                "risk_score": risk_score,
            }
        )

    feature_df = pd.DataFrame(feature_rows).sort_values("risk_score", ascending=False).reset_index(drop=True)

    q10_t1, q50_t1, q90_t1 = predict_target_quantiles(models, "t1_us", x_scaled, target_transforms["t1_us"])
    q10_t2, q50_t2, q90_t2 = predict_target_quantiles(models, "t2_us", x_scaled, target_transforms["t2_us"])

    width_t1 = q90_t1 - q10_t1
    width_t2 = q90_t2 - q10_t2
    combined_unc = (width_t1 - np.mean(width_t1)) / (np.std(width_t1) + 1e-12) + (width_t2 - np.mean(width_t2)) / (np.std(width_t2) + 1e-12)

    uncertain_df = pd.DataFrame(
        {
            "row_index": data["row_index"],
            "design_id": data["design_id"],
            "split": split,
            "t1_target_us": y_eval_all[:, 0],
            "t2_target_us": y_eval_all[:, 1],
            "pred_t1_p10_us": q10_t1,
            "pred_t1_p50_us": q50_t1,
            "pred_t1_p90_us": q90_t1,
            "pred_t2_p10_us": q10_t2,
            "pred_t2_p50_us": q50_t2,
            "pred_t2_p90_us": q90_t2,
            "uncertainty_width_t1": width_t1,
            "uncertainty_width_t2": width_t2,
            "combined_uncertainty_score": combined_unc,
        }
    ).sort_values("combined_uncertainty_score", ascending=False)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(outdir / "phase4_feature_risk_report.csv", index=False)
    uncertain_df.head(max(10, args.uncertain_top_n)).to_csv(outdir / "phase4_high_uncertainty_candidates.csv", index=False)

    bundle = {
        "version": 1,
        "type": "phase4_coherence",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_csv": str(args.single_csv.resolve()),
        "feature_cols": list(FEATURE_COLS),
        "target_cols": list(TARGET_COLS),
        "row_index_all": data["row_index"].astype(int).tolist(),
        "design_id_all": data["design_id"].astype(int).tolist(),
        "train_positions": train_pos.astype(int).tolist(),
        "id_test_positions": id_test_pos.astype(int).tolist(),
        "ood_positions": ood_pos.astype(int).tolist(),
        "scaler_mean": x_mean,
        "scaler_scale": x_scale,
        "models": models,
        "target_transforms": target_transforms,
        "feature_ood_train_scaled": x_scaled[train_pos].astype(np.float32),
        "feature_ood_threshold": feature_ood_threshold,
        "feature_ood_quantile": float(args.feature_ood_quantile),
        "embedding_ref": embedding_ref,
        "feature_medians": {c: float(np.median(data.loc[train_pos, c])) for c in FEATURE_COLS},
        "width_reference": {
            "t1_width_mean": float(np.mean((q90_t1 - q10_t1)[train_pos])),
            "t1_width_p90": float(np.quantile((q90_t1 - q10_t1)[train_pos], 0.90)),
            "t2_width_mean": float(np.mean((q90_t2 - q10_t2)[train_pos])),
            "t2_width_p90": float(np.quantile((q90_t2 - q10_t2)[train_pos], 0.90)),
        },
        "label_stats": label_stats,
        "sample_weight_config": {
            "proxy_weight": float(args.proxy_weight),
            "measured_weight": float(args.measured_weight),
            "measurement_weight_min": float(args.measurement_weight_min),
            "measurement_weight_max": float(args.measurement_weight_max),
            "target_sample_weight_stats": target_sample_weight_stats,
        },
        "label_mode_effective": mode_effective,
        "source_calibration": source_calibration,
        "sample_weight_config": {
            "proxy_weight": float(args.proxy_weight),
            "measured_weight": float(args.measured_weight),
            "measurement_weight_min": float(args.measurement_weight_min),
            "measurement_weight_max": float(args.measurement_weight_max),
            "target_sample_weight_stats": target_sample_weight_stats,
        },
        "ood_thresholds": ood_thresholds,
    }
    joblib.dump(bundle, outdir / "phase4_coherence_bundle.joblib")

    summary = {
        "rows_total": int(len(data)),
        "rows_train": int(len(train_pos)),
        "rows_id_test": int(len(id_test_pos)),
        "rows_ood": int(len(ood_pos)),
        "label_stats": label_stats,
        "label_mode_effective": mode_effective,
        "source_calibration": source_calibration,
        "target_transforms": target_transforms,
        "metrics": {
            "train": metrics_train,
            "id_test": metrics_id,
            "ood": metrics_ood,
        },
        "feature_ood_threshold": feature_ood_threshold,
        "feature_ood_quantile": float(args.feature_ood_quantile),
        "embedding_ood_enabled": bool(embedding_ref.get("enabled", False)),
        "embedding_ood_threshold": float(embedding_ref.get("ood_threshold", np.nan)) if embedding_ref.get("enabled", False) else None,
        "trained_at_utc": bundle["trained_at_utc"],
    }
    (outdir / "phase4_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Phase 4 Coherence Model Trained ===")
    print(f"rows total={summary['rows_total']} train={summary['rows_train']} id_test={summary['rows_id_test']} ood={summary['rows_ood']}")
    print(f"label_mode={mode_effective} measured_rows={label_stats['measurement_rows_matched']}")
    print(
        "ID t1 MAE="
        f"{metrics_id['t1_us']['mae']:.6f} us, "
        "OOD t1 MAE="
        f"{metrics_ood['t1_us']['mae']:.6f} us"
    )
    print(
        "ID t2 log10_MAE="
        f"{metrics_id['t2_us']['log10_mae']:.6f}, "
        "OOD t2 log10_MAE="
        f"{metrics_ood['t2_us']['log10_mae']:.6f}"
    )
    print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
