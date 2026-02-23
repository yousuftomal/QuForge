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
    tier_col = find_first_present(raw.columns, ["measurement_quality_tier", "quality_tier"])
    uncertainty_col = find_first_present(raw.columns, ["uncertainty_inflation_base"])
    quality_mul_col = find_first_present(raw.columns, ["quality_weight_multiplier"])
    fitted_col = find_first_present(raw.columns, ["is_fitted_or_model_row"])
    source_record_col = find_first_present(raw.columns, ["source_record_id"])
    synthetic_col = find_first_present(raw.columns, ["is_synthetic_regularization"])
    anchor_source_col = find_first_present(raw.columns, ["anchor_source_name"])
    anchor_row_col = find_first_present(raw.columns, ["anchor_row_index"])

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
    if tier_col is not None and tier_col not in keep_cols:
        keep_cols.append(tier_col)
    if uncertainty_col is not None and uncertainty_col not in keep_cols:
        keep_cols.append(uncertainty_col)
    if quality_mul_col is not None and quality_mul_col not in keep_cols:
        keep_cols.append(quality_mul_col)
    if fitted_col is not None and fitted_col not in keep_cols:
        keep_cols.append(fitted_col)
    if source_record_col is not None and source_record_col not in keep_cols:
        keep_cols.append(source_record_col)
    if synthetic_col is not None and synthetic_col not in keep_cols:
        keep_cols.append(synthetic_col)
    if anchor_source_col is not None and anchor_source_col not in keep_cols:
        keep_cols.append(anchor_source_col)
    if anchor_row_col is not None and anchor_row_col not in keep_cols:
        keep_cols.append(anchor_row_col)

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
    if tier_col is not None and tier_col != "measurement_quality_tier":
        out = out.rename(columns={tier_col: "measurement_quality_tier"})
    if uncertainty_col is not None and uncertainty_col != "uncertainty_inflation_base":
        out = out.rename(columns={uncertainty_col: "uncertainty_inflation_base"})
    if quality_mul_col is not None and quality_mul_col != "quality_weight_multiplier":
        out = out.rename(columns={quality_mul_col: "quality_weight_multiplier"})
    if fitted_col is not None and fitted_col != "is_fitted_or_model_row":
        out = out.rename(columns={fitted_col: "is_fitted_or_model_row"})
    if source_record_col is not None and source_record_col != "source_record_id":
        out = out.rename(columns={source_record_col: "source_record_id"})
    if synthetic_col is not None and synthetic_col != "is_synthetic_regularization":
        out = out.rename(columns={synthetic_col: "is_synthetic_regularization"})
    if anchor_source_col is not None and anchor_source_col != "anchor_source_name":
        out = out.rename(columns={anchor_source_col: "anchor_source_name"})
    if anchor_row_col is not None and anchor_row_col != "anchor_row_index":
        out = out.rename(columns={anchor_row_col: "anchor_row_index"})

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
    if "measurement_quality_tier" in out.columns:
        out["measurement_quality_tier"] = out["measurement_quality_tier"].fillna("").astype(str)
    if "uncertainty_inflation_base" in out.columns:
        out["uncertainty_inflation_base"] = pd.to_numeric(out["uncertainty_inflation_base"], errors="coerce")
    if "quality_weight_multiplier" in out.columns:
        out["quality_weight_multiplier"] = pd.to_numeric(out["quality_weight_multiplier"], errors="coerce")
    if "is_fitted_or_model_row" in out.columns:
        out["is_fitted_or_model_row"] = out["is_fitted_or_model_row"].fillna(False).astype(bool)
    if "source_record_id" in out.columns:
        out["source_record_id"] = out["source_record_id"].fillna("").astype(str)
    if "is_synthetic_regularization" in out.columns:
        if out["is_synthetic_regularization"].dtype != bool:
            txt = out["is_synthetic_regularization"].fillna("").astype(str).str.strip().str.lower()
            out["is_synthetic_regularization"] = txt.isin({"1", "true", "t", "yes", "y"})
        out["is_synthetic_regularization"] = out["is_synthetic_regularization"].fillna(False).astype(bool)
    else:
        out["is_synthetic_regularization"] = False
    if "anchor_source_name" in out.columns:
        out["anchor_source_name"] = out["anchor_source_name"].fillna("").astype(str)
    else:
        out["anchor_source_name"] = ""
    if "anchor_row_index" in out.columns:
        out["anchor_row_index"] = pd.to_numeric(out["anchor_row_index"], errors="coerce")
    else:
        out["anchor_row_index"] = np.nan

    return out


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    wsum = float(np.sum(w))
    w2sum = float(np.sum(np.square(w)))
    if w2sum <= 1e-18:
        return float(w.size)
    return float((wsum * wsum) / w2sum)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, int]:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if int(mask.sum()) == 0:
        return np.nan, np.nan, 0
    v = vals[mask]
    ww = w[mask]
    wsum = float(np.sum(ww))
    if wsum <= 1e-18:
        return np.nan, np.nan, 0
    mean = float(np.sum(v * ww) / wsum)
    var = float(np.sum(ww * np.square(v - mean)) / wsum)
    return mean, float(np.sqrt(max(var, 0.0))), int(mask.sum())


def _build_measurement_aggregate(base: pd.DataFrame, measurement_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if measurement_df is None or len(measurement_df) == 0:
        return None, None

    mdf = measurement_df.copy()
    id_col: Optional[str] = None
    if "row_index" in mdf.columns and np.isfinite(pd.to_numeric(mdf["row_index"], errors="coerce")).any():
        id_col = "row_index"
        mdf["row_index"] = pd.to_numeric(mdf["row_index"], errors="coerce")
        mdf = mdf.dropna(subset=["row_index"]).copy()
        mdf["row_index"] = mdf["row_index"].astype(int)
        valid_ids = set(base["row_index"].astype(int).tolist())
        mdf = mdf[mdf["row_index"].isin(valid_ids)].copy()
    elif "design_id" in mdf.columns and np.isfinite(pd.to_numeric(mdf["design_id"], errors="coerce")).any():
        id_col = "design_id"
        mdf["design_id"] = pd.to_numeric(mdf["design_id"], errors="coerce")
        mdf = mdf.dropna(subset=["design_id"]).copy()
        mdf["design_id"] = mdf["design_id"].astype(int)
        valid_ids = set(base["design_id"].astype(int).tolist())
        mdf = mdf[mdf["design_id"].isin(valid_ids)].copy()
    else:
        return None, None

    if mdf.empty:
        return None, id_col

    if "confidence_weight" in mdf.columns:
        raw_w = pd.to_numeric(mdf["confidence_weight"], errors="coerce")
    else:
        raw_w = pd.Series(np.ones(len(mdf), dtype=float), index=mdf.index)
    raw_w = raw_w.fillna(1.0).clip(lower=1e-6)
    mdf["confidence_weight"] = raw_w.to_numpy(dtype=float)

    if "quality_weight_multiplier" in mdf.columns:
        qmul = pd.to_numeric(mdf["quality_weight_multiplier"], errors="coerce").fillna(1.0).clip(lower=0.05, upper=2.0)
    else:
        qmul = pd.Series(np.ones(len(mdf), dtype=float), index=mdf.index)
    mdf["quality_weight_multiplier"] = qmul.to_numpy(dtype=float)
    mdf["effective_weight"] = mdf["confidence_weight"] * mdf["quality_weight_multiplier"]

    if "measurement_quality_tier" not in mdf.columns:
        mdf["measurement_quality_tier"] = ""
    mdf["measurement_quality_tier"] = mdf["measurement_quality_tier"].fillna("").astype(str)

    if "uncertainty_inflation_base" in mdf.columns:
        mdf["uncertainty_inflation_base"] = pd.to_numeric(mdf["uncertainty_inflation_base"], errors="coerce").fillna(1.0)
    else:
        mdf["uncertainty_inflation_base"] = 1.0

    if "measured_t1_us" in mdf.columns:
        mdf["measured_t1_us"] = pd.to_numeric(mdf["measured_t1_us"], errors="coerce")
    else:
        mdf["measured_t1_us"] = np.nan
    if "measured_t2_us" in mdf.columns:
        mdf["measured_t2_us"] = pd.to_numeric(mdf["measured_t2_us"], errors="coerce")
    else:
        mdf["measured_t2_us"] = np.nan
    if "source_name" in mdf.columns:
        mdf["source_name"] = mdf["source_name"].fillna("").astype(str)
    else:
        mdf["source_name"] = ""
    if "anchor_source_name" in mdf.columns:
        mdf["anchor_source_name"] = mdf["anchor_source_name"].fillna("").astype(str)
    else:
        mdf["anchor_source_name"] = ""
    if "is_synthetic_regularization" in mdf.columns:
        if mdf["is_synthetic_regularization"].dtype != bool:
            txt = mdf["is_synthetic_regularization"].fillna("").astype(str).str.strip().str.lower()
            mdf["is_synthetic_regularization"] = txt.isin({"1", "true", "t", "yes", "y"})
        mdf["is_synthetic_regularization"] = mdf["is_synthetic_regularization"].fillna(False).astype(bool)
    else:
        mdf["is_synthetic_regularization"] = False
    if "anchor_row_index" in mdf.columns:
        mdf["anchor_row_index"] = pd.to_numeric(mdf["anchor_row_index"], errors="coerce")
    else:
        mdf["anchor_row_index"] = np.nan

    rows: List[Dict[str, object]] = []
    grouped = mdf.groupby(id_col, sort=False, dropna=False)
    for gid, g in grouped:
        weights = pd.to_numeric(g["effective_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        t1 = pd.to_numeric(g["measured_t1_us"], errors="coerce").to_numpy(dtype=float)
        t2 = pd.to_numeric(g["measured_t2_us"], errors="coerce").to_numpy(dtype=float)

        t1_mean, t1_std, t1_count = _weighted_mean_std(t1, weights)
        t2_mean, t2_std, t2_count = _weighted_mean_std(t2, weights)
        eff_n = _effective_sample_size(weights)

        src_w = g.groupby("source_name")["effective_weight"].sum().sort_values(ascending=False)
        dominant_source = str(src_w.index[0]) if len(src_w) > 0 else ""
        synthetic_mask = g["is_synthetic_regularization"].fillna(False).astype(bool).to_numpy(dtype=bool)
        direct_mask = ~synthetic_mask
        direct_count = int(np.sum(direct_mask))
        synthetic_count = int(np.sum(synthetic_mask))

        dominant_anchor_source = ""
        if synthetic_count > 0:
            synth_sources = g.loc[synthetic_mask, "anchor_source_name"].fillna("").astype(str)
            if len(synth_sources) > 0:
                anchor_counts = synth_sources.value_counts()
                if len(anchor_counts) > 0:
                    dominant_anchor_source = str(anchor_counts.index[0])
        anchor_group_row_index = int(gid) if id_col == "row_index" else -1
        if synthetic_count > 0 and "anchor_row_index" in g.columns:
            anchor_rows = pd.to_numeric(g.loc[synthetic_mask, "anchor_row_index"], errors="coerce").dropna().astype(int)
            if len(anchor_rows) > 0:
                anchor_group_row_index = int(anchor_rows.value_counts().index[0])

        tier = g["measurement_quality_tier"].fillna("").astype(str)
        weak_frac = float(np.mean(tier == "weak_source")) if len(g) > 0 else 0.0
        trace_frac = float(np.mean(tier == "tracefit_or_model")) if len(g) > 0 else 0.0
        synth_frac = float(np.mean(synthetic_mask)) if len(g) > 0 else 0.0
        direct_t1_count = int(np.sum(np.isfinite(t1) & direct_mask))
        direct_t2_count = int(np.sum(np.isfinite(t2) & direct_mask))

        rows.append(
            {
                id_col: int(gid),
                "measured_t1_us": t1_mean,
                "measured_t2_us": t2_mean,
                "measured_t1_std_us": t1_std,
                "measured_t2_std_us": t2_std,
                "measured_t1_count": int(t1_count),
                "measured_t2_count": int(t2_count),
                "measurement_count": int(len(g)),
                "measurement_effective_count": float(eff_n),
                "measurement_confidence_weight": float(np.nanmean(weights)) if np.isfinite(weights).any() else 1.0,
                "measurement_source_name": dominant_source,
                "anchor_source_name": dominant_anchor_source,
                "anchor_group_row_index": int(anchor_group_row_index),
                "weak_source_fraction": weak_frac,
                "tracefit_fraction": trace_frac,
                "synthetic_fraction": synth_frac,
                "direct_measurement_count": direct_count,
                "synthetic_measurement_count": synthetic_count,
                "direct_t1_count": direct_t1_count,
                "direct_t2_count": direct_t2_count,
                "uncertainty_inflation_base": float(np.nanmean(pd.to_numeric(g["uncertainty_inflation_base"], errors="coerce").fillna(1.0))),
            }
        )

    agg = pd.DataFrame(rows)
    return agg, id_col


def _canonicalize_measured_by_source(
    values: np.ndarray,
    sources: np.ndarray,
    min_rows: int,
    max_abs_shift_log10: float,
    calibration_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = values.copy()
    values_f = np.asarray(values, dtype=float)
    sources_s = pd.Series(sources).fillna("").astype(str).to_numpy(dtype=object)

    valid = np.isfinite(values_f) & (values_f > 0)
    if calibration_mask is not None:
        calib_mask = np.asarray(calibration_mask, dtype=bool)
        if calib_mask.shape[0] == valid.shape[0]:
            valid = valid & calib_mask
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
    repeated_weight_power: float,
    synthetic_label_blend: float,
    residual_transfer_enable: bool,
    residual_transfer_k: int,
    residual_transfer_tau: float,
    residual_transfer_strength: float,
    residual_transfer_max_abs_log10: float,
    uncertainty_distance_scale: float,
    uncertainty_distance_gain: float,
    uncertainty_max_factor: float,
) -> Tuple[pd.DataFrame, Dict[str, object], str, Dict[str, object]]:
    proxy_t1 = base["t1_estimate_us"].to_numpy(dtype=float)
    proxy_t2 = base["t2_estimate_us"].to_numpy(dtype=float)

    measured_t1 = np.full(len(base), np.nan, dtype=float)
    measured_t2 = np.full(len(base), np.nan, dtype=float)
    measured_w = np.full(len(base), np.nan, dtype=float)
    measured_source = np.array(["" for _ in range(len(base))], dtype=object)
    anchor_source = np.array(["" for _ in range(len(base))], dtype=object)
    anchor_group_row_index = np.full(len(base), np.nan, dtype=float)
    measured_eff_n = np.full(len(base), np.nan, dtype=float)
    weak_source_fraction = np.zeros(len(base), dtype=float)
    tracefit_fraction = np.zeros(len(base), dtype=float)
    synthetic_fraction = np.zeros(len(base), dtype=float)
    measurement_count = np.zeros(len(base), dtype=int)
    direct_measurement_count = np.zeros(len(base), dtype=int)
    synthetic_measurement_count = np.zeros(len(base), dtype=int)
    direct_t1_count = np.zeros(len(base), dtype=int)
    direct_t2_count = np.zeros(len(base), dtype=int)
    uncertainty_base = np.ones(len(base), dtype=float)

    measured_rows = 0
    aggregate_df, id_col = _build_measurement_aggregate(base, measurement_df)
    if aggregate_df is not None and id_col is not None and len(aggregate_df) > 0:
        m = aggregate_df.set_index(id_col)
        if id_col == "row_index":
            base_key = base["row_index"]
        else:
            base_key = base["design_id"]

        measured_t1 = base_key.map(m.get("measured_t1_us")).to_numpy(dtype=float)
        measured_t2 = base_key.map(m.get("measured_t2_us")).to_numpy(dtype=float)
        measured_w = base_key.map(m.get("measurement_confidence_weight")).to_numpy(dtype=float)
        measured_source = base_key.map(m.get("measurement_source_name")).fillna("").astype(str).to_numpy(dtype=object)
        anchor_source = base_key.map(m.get("anchor_source_name")).fillna("").astype(str).to_numpy(dtype=object)
        anchor_group_row_index = base_key.map(m.get("anchor_group_row_index")).to_numpy(dtype=float)
        measured_eff_n = base_key.map(m.get("measurement_effective_count")).to_numpy(dtype=float)
        weak_source_fraction = base_key.map(m.get("weak_source_fraction")).fillna(0.0).to_numpy(dtype=float)
        tracefit_fraction = base_key.map(m.get("tracefit_fraction")).fillna(0.0).to_numpy(dtype=float)
        synthetic_fraction = base_key.map(m.get("synthetic_fraction")).fillna(0.0).to_numpy(dtype=float)
        measurement_count = base_key.map(m.get("measurement_count")).fillna(0).to_numpy(dtype=int)
        direct_measurement_count = base_key.map(m.get("direct_measurement_count")).fillna(0).to_numpy(dtype=int)
        synthetic_measurement_count = base_key.map(m.get("synthetic_measurement_count")).fillna(0).to_numpy(dtype=int)
        direct_t1_count = base_key.map(m.get("direct_t1_count")).fillna(0).to_numpy(dtype=int)
        direct_t2_count = base_key.map(m.get("direct_t2_count")).fillna(0).to_numpy(dtype=int)
        uncertainty_base = base_key.map(m.get("uncertainty_inflation_base")).fillna(1.0).to_numpy(dtype=float)
        measured_rows = int(np.sum(np.isfinite(measured_t1) | np.isfinite(measured_t2)))

    measured_t1_canon, t1_cal = _canonicalize_measured_by_source(
        values=measured_t1,
        sources=measured_source,
        min_rows=source_calibration_min_rows,
        max_abs_shift_log10=source_calibration_max_shift_log10,
        calibration_mask=(direct_t1_count > 0),
    )
    measured_t2_canon, t2_cal = _canonicalize_measured_by_source(
        values=measured_t2,
        sources=measured_source,
        min_rows=source_calibration_min_rows,
        max_abs_shift_log10=source_calibration_max_shift_log10,
        calibration_mask=(direct_t2_count > 0),
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

    def apply_residual_transfer(
        proxy_values: np.ndarray,
        measured_values: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        corrected = proxy_values.copy()
        blend = np.zeros(len(proxy_values), dtype=float)
        min_dist = np.full(len(proxy_values), np.inf, dtype=float)
        if not residual_transfer_enable:
            return corrected, blend, min_dist

        anchor_mask = (
            np.isfinite(proxy_values)
            & (proxy_values > 0)
            & np.isfinite(measured_values)
            & (measured_values > 0)
        )
        n_anchor = int(anchor_mask.sum())
        if n_anchor < 2:
            return corrected, blend, min_dist

        feat = base.loc[:, FEATURE_COLS].to_numpy(dtype=float)
        mu = np.nanmean(feat, axis=0)
        sigma = np.nanstd(feat, axis=0)
        sigma = np.where(sigma <= 1e-12, 1.0, sigma)
        z_all = (feat - mu[None, :]) / sigma[None, :]

        z_anchor = z_all[anchor_mask]
        residual_anchor = np.log10(np.maximum(measured_values[anchor_mask], 1e-18)) - np.log10(np.maximum(proxy_values[anchor_mask], 1e-18))
        anchor_weight = np.where(np.isfinite(sample_weight[anchor_mask]), sample_weight[anchor_mask], 1.0).astype(float)
        anchor_weight = np.clip(anchor_weight, 1e-6, 1e6)

        k_eff = int(max(1, min(int(residual_transfer_k), n_anchor)))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(z_anchor)
        dist, idx = nn.kneighbors(z_all, n_neighbors=k_eff)

        tau = float(max(residual_transfer_tau, 1e-6))
        kernel = np.exp(-np.square(dist / tau))
        anchor_w = anchor_weight[idx]
        weighted = kernel * anchor_w
        den = np.sum(weighted, axis=1) + 1e-12
        res = np.sum(weighted * residual_anchor[idx], axis=1) / den
        res = np.clip(res, -abs(residual_transfer_max_abs_log10), abs(residual_transfer_max_abs_log10))

        local_strength = np.clip(np.sum(kernel, axis=1) / float(k_eff), 0.0, 1.0)
        blend = np.clip(float(residual_transfer_strength) * local_strength, 0.0, 1.0)
        blend[anchor_mask] = 0.0

        corrected_log = np.log10(np.maximum(proxy_values, 1e-18)) + blend * res
        corrected = np.power(10.0, corrected_log)
        min_dist = dist[:, 0]
        return corrected, blend, min_dist

    proxy_t1_corrected, proxy_blend_t1, anchor_min_dist_t1 = apply_residual_transfer(
        proxy_values=proxy_t1,
        measured_values=measured_t1_canon,
        sample_weight=np.where(np.isfinite(measured_w), measured_w, 1.0),
    )
    proxy_t2_corrected, proxy_blend_t2, anchor_min_dist_t2 = apply_residual_transfer(
        proxy_values=proxy_t2,
        measured_values=measured_t2_canon,
        sample_weight=np.where(np.isfinite(measured_w), measured_w, 1.0),
    )
    proxy_transfer_blend = np.maximum(proxy_blend_t1, proxy_blend_t2)
    anchor_min_dist = np.minimum(anchor_min_dist_t1, anchor_min_dist_t2)
    if np.isfinite(anchor_min_dist).any():
        dist_scale = float(uncertainty_distance_scale) if uncertainty_distance_scale > 0 else float(np.nanmedian(anchor_min_dist[np.isfinite(anchor_min_dist)]))
    else:
        dist_scale = 1.0
    dist_scale = float(max(dist_scale, 1e-6))
    dist_ratio = np.where(np.isfinite(anchor_min_dist), anchor_min_dist / dist_scale, 3.0)
    dist_factor = 1.0 + float(max(0.0, uncertainty_distance_gain)) * np.clip(dist_ratio, 0.0, 3.0)
    evidence_factor = np.maximum(
        1.0,
        uncertainty_base + 0.25 * weak_source_fraction + 0.60 * tracefit_fraction + 0.85 * synthetic_fraction,
    )
    uncertainty_inflation = np.clip(np.maximum(dist_factor, evidence_factor), 1.0, float(max(1.0, uncertainty_max_factor)))

    if mode == "proxy":
        t1 = proxy_t1_corrected.copy()
        t2 = proxy_t2_corrected.copy()
    elif mode == "hybrid":
        synth_blend = float(np.clip(synthetic_label_blend, 0.0, 1.0))
        direct_t1_mask = (direct_t1_count > 0) & np.isfinite(measured_t1_canon)
        direct_t2_mask = (direct_t2_count > 0) & np.isfinite(measured_t2_canon)
        synth_t1_mask = (~direct_t1_mask) & np.isfinite(measured_t1_canon)
        synth_t2_mask = (~direct_t2_mask) & np.isfinite(measured_t2_canon)

        t1 = proxy_t1_corrected.copy()
        t2 = proxy_t2_corrected.copy()
        t1[direct_t1_mask] = measured_t1_canon[direct_t1_mask]
        t2[direct_t2_mask] = measured_t2_canon[direct_t2_mask]
        t1[synth_t1_mask] = synth_blend * measured_t1_canon[synth_t1_mask] + (1.0 - synth_blend) * proxy_t1_corrected[synth_t1_mask]
        t2[synth_t2_mask] = synth_blend * measured_t2_canon[synth_t2_mask] + (1.0 - synth_blend) * proxy_t2_corrected[synth_t2_mask]
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
    out["has_direct_measured_t1"] = (direct_t1_count > 0) & np.isfinite(measured_t1)
    out["has_direct_measured_t2"] = (direct_t2_count > 0) & np.isfinite(measured_t2)
    out["has_synthetic_only_t1"] = out["has_measured_t1"] & (~out["has_direct_measured_t1"])
    out["has_synthetic_only_t2"] = out["has_measured_t2"] & (~out["has_direct_measured_t2"])
    base_measured_weight = np.where(np.isfinite(measured_w), measured_w, 1.0)
    repeated_scale = np.power(np.maximum(np.where(np.isfinite(measured_eff_n), measured_eff_n, 1.0), 1.0), float(max(0.0, repeated_weight_power)))
    quality_scale = np.exp(-(0.7 * weak_source_fraction + 1.3 * tracefit_fraction + 1.6 * synthetic_fraction))
    out["measurement_confidence_weight"] = base_measured_weight * repeated_scale * quality_scale
    out["measurement_confidence_weight"] = np.clip(out["measurement_confidence_weight"], 1e-3, 1e3)
    out["measurement_source_name"] = pd.Series(measured_source).fillna("").astype(str).to_numpy(dtype=object)
    out["anchor_source_name"] = pd.Series(anchor_source).fillna("").astype(str).to_numpy(dtype=object)
    out["anchor_group_row_index"] = np.where(
        np.isfinite(anchor_group_row_index),
        anchor_group_row_index,
        out["row_index"].to_numpy(dtype=float),
    )
    out["measurement_effective_count"] = np.where(np.isfinite(measured_eff_n), measured_eff_n, 0.0)
    out["measurement_count"] = measurement_count
    out["direct_measurement_count"] = direct_measurement_count
    out["synthetic_measurement_count"] = synthetic_measurement_count
    out["weak_source_fraction"] = weak_source_fraction
    out["tracefit_fraction"] = tracefit_fraction
    out["synthetic_measurement_fraction"] = synthetic_fraction
    out["proxy_transfer_blend"] = proxy_transfer_blend
    out["anchor_min_distance"] = anchor_min_dist
    out["uncertainty_inflation_factor"] = uncertainty_inflation

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[*FEATURE_COLS, "t1_us", "t2_us"]).reset_index(drop=True)

    measured_conf = out.loc[(out["has_measured_t1"] | out["has_measured_t2"]), "measurement_confidence_weight"]

    stats = {
        "mode_requested": label_mode,
        "mode_effective": mode,
        "measurement_rows_matched": measured_rows,
        "measurement_rows_direct_matched": int(np.sum((direct_t1_count > 0) | (direct_t2_count > 0))),
        "measurement_rows_synthetic_only_matched": int(np.sum(((direct_t1_count <= 0) & np.isfinite(measured_t1)) | ((direct_t2_count <= 0) & np.isfinite(measured_t2)))),
        "rows_final": int(len(out)),
        "rows_with_measured_t1": int(np.sum(np.isfinite(measured_t1))),
        "rows_with_measured_t2": int(np.sum(np.isfinite(measured_t2))),
        "measurement_aggregate": {
            "raw_rows": int(len(measurement_df)) if measurement_df is not None else 0,
            "aggregated_rows": int(len(aggregate_df)) if aggregate_df is not None else 0,
            "id_mode": id_col,
            "mean_repeated_measurement_count": float(np.nanmean(out.loc[(out["has_measured_t1"] | out["has_measured_t2"]), "measurement_count"])) if int((out["has_measured_t1"] | out["has_measured_t2"]).sum()) > 0 else 0.0,
            "mean_effective_sample_count": float(np.nanmean(out.loc[(out["has_measured_t1"] | out["has_measured_t2"]), "measurement_effective_count"])) if int((out["has_measured_t1"] | out["has_measured_t2"]).sum()) > 0 else 0.0,
            "rows_with_direct_measurements": int((out["has_direct_measured_t1"] | out["has_direct_measured_t2"]).sum()),
            "rows_with_synthetic_only_measurements": int((out["has_synthetic_only_t1"] | out["has_synthetic_only_t2"]).sum()),
            "mean_synthetic_measurement_fraction": float(np.nanmean(out["synthetic_measurement_fraction"])) if len(out) > 0 else 0.0,
        },
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
        "anchor_source_counts": {
            str(k): int(v)
            for k, v in out.loc[(out["has_synthetic_only_t1"] | out["has_synthetic_only_t2"]), "anchor_source_name"].value_counts().to_dict().items()
            if str(k)
        },
        "source_calibration": source_calibration,
        "residual_transfer": {
            "enabled": bool(residual_transfer_enable),
            "k_neighbors": int(residual_transfer_k),
            "tau": float(residual_transfer_tau),
            "strength": float(residual_transfer_strength),
            "max_abs_log10": float(residual_transfer_max_abs_log10),
            "synthetic_label_blend": float(np.clip(synthetic_label_blend, 0.0, 1.0)),
            "uncertainty_distance_scale": float(dist_scale),
            "uncertainty_distance_gain": float(max(0.0, uncertainty_distance_gain)),
            "uncertainty_max_factor": float(max(1.0, uncertainty_max_factor)),
            "mean_proxy_transfer_blend": float(np.nanmean(out["proxy_transfer_blend"])) if len(out) > 0 else 0.0,
            "mean_uncertainty_inflation_factor": float(np.nanmean(out["uncertainty_inflation_factor"])) if len(out) > 0 else 1.0,
        },
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


def grouped_train_test_split(
    positions: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=int)
    grp = np.asarray(groups)
    if pos.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    if grp.shape[0] != pos.shape[0]:
        return train_test_split(pos, test_size=test_size, random_state=random_state, shuffle=True)

    grp_s = pd.Series(grp).fillna(-1).astype(int).to_numpy(dtype=int)
    uniq = np.unique(grp_s)
    if uniq.size < 2:
        return train_test_split(pos, test_size=test_size, random_state=random_state, shuffle=True)

    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(uniq.size)
    uniq_perm = uniq[perm]

    n_test_groups = int(max(1, round(float(test_size) * float(uniq_perm.size))))
    n_test_groups = int(min(max(1, n_test_groups), max(1, uniq_perm.size - 1)))
    test_groups = set(uniq_perm[:n_test_groups].tolist())
    test_mask = np.array([g in test_groups for g in grp_s], dtype=bool)
    test_pos = pos[test_mask]
    train_pos = pos[~test_mask]
    if train_pos.size == 0 or test_pos.size == 0:
        return train_test_split(pos, test_size=test_size, random_state=random_state, shuffle=True)
    return train_pos.astype(int), test_pos.astype(int)


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
    parser.add_argument("--repeated-weight-power", type=float, default=0.5, help="Diminishing-return exponent for repeated measurements")
    parser.add_argument("--synthetic-label-blend", type=float, default=0.35, help="Blend factor for synthetic-only labels in hybrid mode")
    parser.add_argument("--synthetic-regularization-weight", type=float, default=0.75, help="Sample-weight multiplier for synthetic-only rows")

    parser.add_argument("--residual-transfer-disable", action="store_true", help="Disable anchor-conditioned residual transfer")
    parser.add_argument("--residual-transfer-k", type=int, default=12)
    parser.add_argument("--residual-transfer-tau", type=float, default=1.25)
    parser.add_argument("--residual-transfer-strength", type=float, default=0.35)
    parser.add_argument("--residual-transfer-max-abs-log10", type=float, default=0.35)
    parser.add_argument("--uncertainty-distance-scale", type=float, default=0.0, help="If <=0, inferred from anchor distances")
    parser.add_argument("--uncertainty-distance-gain", type=float, default=0.45)
    parser.add_argument("--uncertainty-max-factor", type=float, default=2.50)

    parser.add_argument("--id-test-size", type=float, default=0.2)
    parser.add_argument("--disable-grouped-anchor-split", action="store_true", help="Disable group-wise split by anchor/design IDs")
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
    if "row_index" in df_raw.columns:
        row_idx = pd.to_numeric(df_raw["row_index"], errors="coerce")
        if np.isfinite(row_idx).any():
            base["row_index"] = row_idx.to_numpy(dtype=float)
        else:
            base["row_index"] = df_raw.index.to_numpy(dtype=float)
    else:
        base["row_index"] = df_raw.index.to_numpy(dtype=float)
    base["row_index"] = pd.to_numeric(base["row_index"], errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"]).reset_index(drop=True)
    base["row_index"] = base["row_index"].astype(int)

    measurement_df = load_measurement_df(args.measurement_csv)
    data, label_stats, mode_effective, source_calibration = derive_targets(
        base,
        measurement_df,
        args.label_mode,
        source_calibration_min_rows=args.source_calibration_min_rows,
        source_calibration_max_shift_log10=args.source_calibration_max_shift_log10,
        repeated_weight_power=args.repeated_weight_power,
        synthetic_label_blend=args.synthetic_label_blend,
        residual_transfer_enable=not args.residual_transfer_disable,
        residual_transfer_k=args.residual_transfer_k,
        residual_transfer_tau=args.residual_transfer_tau,
        residual_transfer_strength=args.residual_transfer_strength,
        residual_transfer_max_abs_log10=args.residual_transfer_max_abs_log10,
        uncertainty_distance_scale=args.uncertainty_distance_scale,
        uncertainty_distance_gain=args.uncertainty_distance_gain,
        uncertainty_max_factor=args.uncertainty_max_factor,
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

    if (not args.disable_grouped_anchor_split) and ("anchor_group_row_index" in data.columns):
        groups_in = data.loc[in_pos, "anchor_group_row_index"].to_numpy(dtype=float)
        train_pos, id_test_pos = grouped_train_test_split(
            positions=in_pos,
            groups=groups_in,
            test_size=args.id_test_size,
            random_state=args.random_state,
        )
    else:
        train_pos, id_test_pos = train_test_split(
            in_pos,
            test_size=args.id_test_size,
            random_state=args.random_state,
            shuffle=True,
        )
    grouped_split_enabled = bool((not args.disable_grouped_anchor_split) and ("anchor_group_row_index" in data.columns))
    if "anchor_group_row_index" in data.columns:
        grp_all = pd.to_numeric(data["anchor_group_row_index"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    else:
        grp_all = data["row_index"].to_numpy(dtype=int)
    train_group_count = int(np.unique(grp_all[train_pos]).size) if len(train_pos) > 0 else 0
    id_group_count = int(np.unique(grp_all[id_test_pos]).size) if len(id_test_pos) > 0 else 0

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
            has_direct = data["has_direct_measured_t1"].to_numpy(dtype=bool)
        else:
            has_measured = data["has_measured_t2"].to_numpy(dtype=bool)
            has_direct = data["has_direct_measured_t2"].to_numpy(dtype=bool)
        has_synth_only = has_measured & (~has_direct)

        conf = data["measurement_confidence_weight"].to_numpy(dtype=float)
        conf = np.clip(conf, args.measurement_weight_min, args.measurement_weight_max)
        proxy_blend = np.clip(data["proxy_transfer_blend"].to_numpy(dtype=float), 0.0, 1.0)
        proxy_weight_all = args.proxy_weight * (1.0 + 0.50 * proxy_blend)
        sample_weight_all = proxy_weight_all.astype(float)
        sample_weight_all[has_synth_only] = float(args.synthetic_regularization_weight) * conf[has_synth_only]
        sample_weight_all[has_direct] = float(args.measured_weight) * conf[has_direct]
        sample_weight_all = np.maximum(sample_weight_all, 1e-6)
        sample_weight_train = sample_weight_all[train_pos]

        target_sample_weight_stats[target] = {
            "train_min": float(np.min(sample_weight_train)),
            "train_median": float(np.median(sample_weight_train)),
            "train_max": float(np.max(sample_weight_train)),
            "train_mean": float(np.mean(sample_weight_train)),
            "train_fraction_measured": float(np.mean(has_measured[train_pos])),
            "train_fraction_direct_measured": float(np.mean(has_direct[train_pos])),
            "train_fraction_synthetic_only": float(np.mean(has_synth_only[train_pos])),
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

    inflation_all = np.clip(data["uncertainty_inflation_factor"].to_numpy(dtype=float), 1.0, np.inf)

    def apply_uncertainty_inflation(
        q10: np.ndarray,
        q50: np.ndarray,
        q90: np.ndarray,
        factors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.clip(np.asarray(factors, dtype=float), 1.0, np.inf)
        lo = np.maximum(q50 - q10, 1e-18)
        hi = np.maximum(q90 - q50, 1e-18)
        q10_i = q50 - lo * f
        q90_i = q50 + hi * f
        stacked = np.vstack([q10_i, q50, q90_i]).T
        stacked.sort(axis=1)
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]

    def eval_split(positions: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        x = x_scaled[positions]
        infl = inflation_all[positions]
        for t_idx, target in enumerate(TARGET_COLS):
            y = y_eval_all[positions, t_idx]
            q10, q50, q90 = predict_target_quantiles(models, target, x, target_transforms[target])
            q10, q50, q90 = apply_uncertainty_inflation(q10, q50, q90, infl)
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
    q10_t1, q50_t1, q90_t1 = apply_uncertainty_inflation(q10_t1, q50_t1, q90_t1, inflation_all)
    q10_t2, q50_t2, q90_t2 = apply_uncertainty_inflation(q10_t2, q50_t2, q90_t2, inflation_all)

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
            "uncertainty_inflation_factor": inflation_all,
            "proxy_transfer_blend": data["proxy_transfer_blend"].to_numpy(dtype=float),
            "anchor_min_distance": data["anchor_min_distance"].to_numpy(dtype=float),
            "combined_uncertainty_score": combined_unc,
        }
    ).sort_values("combined_uncertainty_score", ascending=False)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(outdir / "phase4_feature_risk_report.csv", index=False)
    uncertain_df.head(max(10, args.uncertain_top_n)).to_csv(outdir / "phase4_high_uncertainty_candidates.csv", index=False)

    measured_anchor_mask = data["has_measured_t1"].to_numpy(dtype=bool) | data["has_measured_t2"].to_numpy(dtype=bool)
    measured_anchor_features_scaled = x_scaled[measured_anchor_mask].astype(np.float32)

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
        "measured_anchor_feature_scaled": measured_anchor_features_scaled,
        "target_reference": {
            "t1_us": y_eval_all[:, 0].astype(float).tolist(),
            "t2_us": y_eval_all[:, 1].astype(float).tolist(),
            "uncertainty_inflation_factor": inflation_all.astype(float).tolist(),
        },
        "uncertainty_inflation_config": {
            "distance_scale": float(max(label_stats.get("residual_transfer", {}).get("uncertainty_distance_scale", args.uncertainty_distance_scale or 1.0), 1e-6)),
            "distance_gain": float(label_stats.get("residual_transfer", {}).get("uncertainty_distance_gain", max(0.0, args.uncertainty_distance_gain))),
            "max_factor": float(label_stats.get("residual_transfer", {}).get("uncertainty_max_factor", max(1.0, args.uncertainty_max_factor))),
            "mean_train_factor": float(np.mean(inflation_all[train_pos])) if len(train_pos) > 0 else 1.0,
        },
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
            "synthetic_regularization_weight": float(args.synthetic_regularization_weight),
            "measurement_weight_min": float(args.measurement_weight_min),
            "measurement_weight_max": float(args.measurement_weight_max),
            "synthetic_label_blend": float(np.clip(args.synthetic_label_blend, 0.0, 1.0)),
            "target_sample_weight_stats": target_sample_weight_stats,
        },
        "label_mode_effective": mode_effective,
        "source_calibration": source_calibration,
        "ood_thresholds": ood_thresholds,
        "split_config": {
            "grouped_anchor_split_enabled": grouped_split_enabled,
            "train_group_count": train_group_count,
            "id_test_group_count": id_group_count,
        },
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
        "mean_train_uncertainty_inflation_factor": float(np.mean(inflation_all[train_pos])) if len(train_pos) > 0 else 1.0,
        "grouped_anchor_split_enabled": grouped_split_enabled,
        "train_group_count": train_group_count,
        "id_test_group_count": id_group_count,
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
