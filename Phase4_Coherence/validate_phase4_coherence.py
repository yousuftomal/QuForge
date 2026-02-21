#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
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


def encode_geom_in_batches(model: TwinEncoder, x: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    out: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(dtype=torch.float32)
            zb = model.encode_geom(xb).cpu().numpy()
            out.append(zb)
    return l2_normalize(np.vstack(out))


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


def invert_target_transform(y_model: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    y = y_model.copy()
    if bool(transform["use_log"]):
        y = np.power(10.0, y)
    lo = float(transform["clip_low"])
    hi = float(transform["clip_high"])
    return np.clip(y, lo, hi)


def predict_target_quantiles(models: Dict[str, object], target: str, x: np.ndarray, transform: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10_m = models[f"{target}_q10"].predict(x)
    q50_m = models[f"{target}_q50"].predict(x)
    q90_m = models[f"{target}_q90"].predict(x)
    q10 = invert_target_transform(q10_m, transform)
    q50 = invert_target_transform(q50_m, transform)
    q90 = invert_target_transform(q90_m, transform)
    stacked = np.vstack([q10, q50, q90]).T
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


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
    out["measurement_source_name"] = pd.Series(measured_source).fillna("").astype(str).to_numpy(dtype=object)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    return out


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Phase 4 coherence predictor")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--measurement-csv", type=Path, default=root / "Dataset" / "measurement_dataset.csv")
    parser.add_argument("--bundle-path", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "phase4_coherence_bundle.joblib")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)

    feature_cols = list(bundle["feature_cols"])
    target_cols = list(bundle["target_cols"])
    target_transforms = dict(bundle["target_transforms"])
    row_index_all = list(bundle["row_index_all"])

    df_raw = pd.read_csv(args.single_csv)
    required = ["design_id", *feature_cols, "t1_estimate_us", "t2_estimate_us"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    base = df_raw[["design_id", *feature_cols, "t1_estimate_us", "t2_estimate_us"]].copy()
    base["row_index"] = df_raw.index.to_numpy()
    base = base.replace([np.inf, -np.inf], np.nan).dropna()

    label_mode = str(bundle.get("label_mode_effective", "proxy"))
    source_calibration = bundle.get("source_calibration", {})
    measurement_df = load_measurement_df(args.measurement_csv)
    data = derive_targets(base, measurement_df, label_mode, source_calibration=source_calibration)

    frame = data.set_index("row_index")
    missing_rows = [idx for idx in row_index_all if idx not in frame.index]
    if missing_rows:
        raise SystemExit("Validation dataset mismatch with training rows")
    data = frame.loc[row_index_all].reset_index()

    x_raw = data.loc[:, feature_cols].to_numpy(dtype=float)
    y_raw = data.loc[:, target_cols].to_numpy(dtype=float)

    y_eval = np.zeros_like(y_raw)
    for i, target in enumerate(target_cols):
        lo = float(target_transforms[target]["clip_low"])
        hi = float(target_transforms[target]["clip_high"])
        y_eval[:, i] = np.clip(y_raw[:, i], lo, hi)

    x_mean = np.asarray(bundle["scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["scaler_scale"], dtype=np.float32)
    x_scaled = ((x_raw - x_mean) / x_scale).astype(np.float32)

    train_pos = np.asarray(bundle["train_positions"], dtype=int)
    id_test_pos = np.asarray(bundle["id_test_positions"], dtype=int)
    ood_pos = np.asarray(bundle["ood_positions"], dtype=int)

    models = bundle["models"]

    def eval_split(positions: np.ndarray) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        x = x_scaled[positions]
        for t_idx, target in enumerate(target_cols):
            y = y_eval[positions, t_idx]
            q10, q50, q90 = predict_target_quantiles(models, target, x, target_transforms[target])
            out[target] = quantile_metrics(y, q10, q50, q90)
        return out

    metrics_train = eval_split(train_pos)
    metrics_id = eval_split(id_test_pos)
    metrics_ood = eval_split(ood_pos)

    true_ood = np.zeros(len(data), dtype=int)
    true_ood[ood_pos] = 1
    eval_subset = np.concatenate([id_test_pos, ood_pos])

    feat_ref = np.asarray(bundle["feature_ood_train_scaled"], dtype=np.float32)
    feat_thr = float(bundle["feature_ood_threshold"])
    feat_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    feat_nn.fit(feat_ref)
    feat_dist, _ = feat_nn.kneighbors(x_scaled, n_neighbors=1)
    feat_flag = (feat_dist[:, 0] > feat_thr).astype(int)
    feat_cls = classification_metrics(true_ood[eval_subset], feat_flag[eval_subset])

    emb_metrics: Optional[Dict[str, float]] = None
    combined_metrics: Optional[Dict[str, float]] = None
    emb_flag = np.zeros(len(data), dtype=int)

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

        xg = data.loc[:, geom_cols].to_numpy(dtype=float)
        xg_scaled = ((xg - g_mean) / g_scale).astype(np.float32)
        z = encode_geom_in_batches(model, xg_scaled, batch_size=2048)

        z_nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        z_nn.fit(z_train)
        z_dist, _ = z_nn.kneighbors(z, n_neighbors=1)
        emb_flag = (z_dist[:, 0] > z_thr).astype(int)

        emb_metrics = classification_metrics(true_ood[eval_subset], emb_flag[eval_subset])
        comb_flag = ((feat_flag + emb_flag) > 0).astype(int)
        combined_metrics = classification_metrics(true_ood[eval_subset], comb_flag[eval_subset])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(Path(args.bundle_path).resolve()),
        "single_csv": str(Path(args.single_csv).resolve()),
        "rows": {
            "total": int(len(data)),
            "train": int(len(train_pos)),
            "id_test": int(len(id_test_pos)),
            "ood": int(len(ood_pos)),
        },
        "label_mode_effective": label_mode,
        "target_transforms": target_transforms,
        "metrics": {
            "train": metrics_train,
            "id_test": metrics_id,
            "ood": metrics_ood,
        },
        "ood_detection": {
            "feature_distance": feat_cls,
            "embedding_distance": emb_metrics,
            "combined_or": combined_metrics,
        },
    }

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "phase4_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = []
    md.append("# Phase 4 Validation Report")
    md.append("")
    md.append(f"Generated: {report['generated_at_utc']}")
    md.append("")
    md.append("## Accuracy")
    md.append("")
    md.append(
        f"- ID t1 MAE: {metrics_id['t1_us']['mae']:.6f} us, OOD t1 MAE: {metrics_ood['t1_us']['mae']:.6f} us"
    )
    md.append(
        f"- ID t2 log10_MAE: {metrics_id['t2_us']['log10_mae']:.6f}, OOD t2 log10_MAE: {metrics_ood['t2_us']['log10_mae']:.6f}"
    )
    md.append(
        f"- ID t1 interval coverage: {metrics_id['t1_us']['interval_80_coverage']:.4f}, OOD: {metrics_ood['t1_us']['interval_80_coverage']:.4f}"
    )
    md.append(
        f"- ID t2 interval coverage: {metrics_id['t2_us']['interval_80_coverage']:.4f}, OOD: {metrics_ood['t2_us']['interval_80_coverage']:.4f}"
    )
    md.append("")
    md.append("## OOD Detection")
    md.append("")
    md.append(
        f"- Feature OOD: precision={feat_cls['precision']:.4f}, recall={feat_cls['recall']:.4f}, f1={feat_cls['f1']:.4f}"
    )
    if emb_metrics is not None:
        md.append(
            f"- Embedding OOD: precision={emb_metrics['precision']:.4f}, recall={emb_metrics['recall']:.4f}, f1={emb_metrics['f1']:.4f}"
        )
    if combined_metrics is not None:
        md.append(
            f"- Combined OOD: precision={combined_metrics['precision']:.4f}, recall={combined_metrics['recall']:.4f}, f1={combined_metrics['f1']:.4f}"
        )

    (outdir / "phase4_validation_report.md").write_text("\n".join(md), encoding="utf-8")

    print("=== Phase 4 Validation Complete ===")
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
