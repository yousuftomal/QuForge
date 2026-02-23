#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

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
    parser = argparse.ArgumentParser(
        description=(
            "Generate anchor-conditioned synthetic regularization rows around mapped measured anchors. "
            "Outputs are synthetic-only regularization labels, not direct measured evidence."
        )
    )
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--measurement-csv", type=Path, default=root / "Dataset" / "measurement_dataset_public_bootstrap.csv")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "Dataset" / "measurement_dataset_public_bootstrap_augmented.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=root / "Dataset" / "measurement_dataset_public_bootstrap_augmented.report.json",
    )

    parser.add_argument("--neighbors-per-anchor", type=int, default=24)
    parser.add_argument("--neighbor-candidates", type=int, default=96)
    parser.add_argument("--distance-tau", type=float, default=1.25)
    parser.add_argument("--residual-max-abs-log10", type=float, default=0.35)
    parser.add_argument("--max-anchors", type=int, default=0, help="0 means no cap")
    parser.add_argument("--max-synthetic-per-design", type=int, default=6)

    parser.add_argument("--synthetic-confidence-scale", type=float, default=0.35)
    parser.add_argument("--synthetic-weight-min", type=float, default=0.03)
    parser.add_argument("--synthetic-weight-max", type=float, default=0.35)
    parser.add_argument("--quality-weight-multiplier", type=float, default=0.40)
    parser.add_argument("--uncertainty-inflation-base", type=float, default=1.80)

    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    txt = s.fillna("").astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "t", "yes", "y"})


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) == 0:
        return float("nan")
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    den = float(np.sum(w))
    if den <= 1e-18:
        return float("nan")
    return float(np.sum(v * w) / den)


def _dominant_by_weight(values: pd.Series, weights: np.ndarray) -> str:
    tmp = pd.DataFrame({"value": values.fillna("").astype(str), "weight": weights})
    if tmp.empty:
        return ""
    agg = tmp.groupby("value", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    if agg.empty:
        return ""
    return str(agg.iloc[0]["value"])


def _prepare_single(single_csv: Path) -> pd.DataFrame:
    single = pd.read_csv(single_csv)
    _ensure_columns(single, ["design_id", *FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"])
    if "row_index" in single.columns:
        row_idx = pd.to_numeric(single["row_index"], errors="coerce")
        if np.isfinite(row_idx).any():
            single["row_index"] = row_idx
        else:
            single["row_index"] = np.arange(len(single), dtype=float)
    else:
        single["row_index"] = np.arange(len(single), dtype=float)
    single["row_index"] = pd.to_numeric(single["row_index"], errors="coerce")
    single = single.dropna(subset=["row_index", "design_id", *FEATURE_COLS, "t1_estimate_us", "t2_estimate_us"]).copy()
    single["row_index"] = single["row_index"].astype(int)
    single["design_id"] = pd.to_numeric(single["design_id"], errors="coerce").astype(int)
    single = single.reset_index(drop=True)
    return single


def _prepare_measurement(measurement_csv: Path, single: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(measurement_csv)
    if m.empty:
        return m

    if "row_index" not in m.columns:
        m["row_index"] = np.nan
    m["row_index"] = pd.to_numeric(m["row_index"], errors="coerce")

    if m["row_index"].isna().any() and "design_id" in m.columns:
        did_to_row = (
            single[["design_id", "row_index"]]
            .drop_duplicates(subset=["design_id"], keep="first")
            .set_index("design_id")["row_index"]
            .to_dict()
        )
        did = pd.to_numeric(m["design_id"], errors="coerce")
        fill = did.map(did_to_row)
        m.loc[m["row_index"].isna(), "row_index"] = fill[m["row_index"].isna()]

    m["row_index"] = pd.to_numeric(m["row_index"], errors="coerce")
    m = m.dropna(subset=["row_index"]).copy()
    m["row_index"] = m["row_index"].astype(int)

    valid_rows = set(single["row_index"].astype(int).tolist())
    m = m[m["row_index"].isin(valid_rows)].copy()

    if "design_id" not in m.columns:
        row_to_design = single.set_index("row_index")["design_id"].to_dict()
        m["design_id"] = m["row_index"].map(row_to_design)
    m["design_id"] = pd.to_numeric(m["design_id"], errors="coerce")
    m = m.dropna(subset=["design_id"]).copy()
    m["design_id"] = m["design_id"].astype(int)

    for c in ("measured_t1_us", "measured_t2_us", "confidence_weight", "quality_weight_multiplier"):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    if "source_name" not in m.columns:
        m["source_name"] = ""
    m["source_name"] = m["source_name"].fillna("").astype(str)

    if "source_record_id" not in m.columns:
        m["source_record_id"] = ""
    m["source_record_id"] = m["source_record_id"].fillna("").astype(str)

    if "is_synthetic_regularization" not in m.columns:
        m["is_synthetic_regularization"] = False
    m["is_synthetic_regularization"] = _as_bool_series(m["is_synthetic_regularization"]).astype(bool)

    return m


def _aggregate_anchor_rows(measurement_df: pd.DataFrame) -> pd.DataFrame:
    if measurement_df.empty:
        return pd.DataFrame()

    direct = measurement_df.loc[~measurement_df["is_synthetic_regularization"]].copy()
    if direct.empty:
        return pd.DataFrame()

    if "confidence_weight" in direct.columns:
        conf = pd.to_numeric(direct["confidence_weight"], errors="coerce").fillna(1.0)
    else:
        conf = pd.Series(np.ones(len(direct), dtype=float), index=direct.index)
    if "quality_weight_multiplier" in direct.columns:
        qmul = pd.to_numeric(direct["quality_weight_multiplier"], errors="coerce").fillna(1.0)
    else:
        qmul = pd.Series(np.ones(len(direct), dtype=float), index=direct.index)

    direct["_w"] = np.clip(conf.to_numpy(dtype=float), 1e-6, np.inf) * np.clip(qmul.to_numpy(dtype=float), 0.05, 5.0)

    rows: List[Dict[str, object]] = []
    for rid, g in direct.groupby("row_index", sort=False):
        w = pd.to_numeric(g["_w"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        t1 = pd.to_numeric(g.get("measured_t1_us"), errors="coerce").to_numpy(dtype=float)
        t2 = pd.to_numeric(g.get("measured_t2_us"), errors="coerce").to_numpy(dtype=float)

        t1_mean = _weighted_mean(t1, w)
        t2_mean = _weighted_mean(t2, w)

        if not (np.isfinite(t1_mean) or np.isfinite(t2_mean)):
            continue

        rows.append(
            {
                "row_index": int(rid),
                "design_id": int(pd.to_numeric(g["design_id"], errors="coerce").dropna().iloc[0]),
                "anchor_measured_t1_us": t1_mean,
                "anchor_measured_t2_us": t2_mean,
                "anchor_source_name": _dominant_by_weight(g["source_name"], w),
                "anchor_source_record_id": _dominant_by_weight(g["source_record_id"], w),
                "anchor_direct_rows": int(len(g)),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["row_index"], keep="first").reset_index(drop=True)
    return out


def _build_neighbor_index(single: pd.DataFrame) -> Tuple[NearestNeighbors, np.ndarray, np.ndarray, np.ndarray]:
    feat = single.loc[:, FEATURE_COLS].to_numpy(dtype=float)
    mu = np.nanmean(feat, axis=0)
    sigma = np.nanstd(feat, axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    z = (feat - mu[None, :]) / sigma[None, :]

    nn = NearestNeighbors(n_neighbors=min(len(single), 512), metric="euclidean")
    nn.fit(z)
    return nn, z, mu, sigma


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)

    single = _prepare_single(args.single_csv)
    measurement = _prepare_measurement(args.measurement_csv, single)
    if measurement.empty:
        if args.strict:
            raise SystemExit("Measurement CSV is empty after preprocessing")
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        measurement.to_csv(args.output_csv, index=False)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "no_input_rows",
            "measurement_csv": str(args.measurement_csv.resolve()),
            "output_csv": str(args.output_csv.resolve()),
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("No measurement rows available; wrote passthrough output")
        return 0

    anchors = _aggregate_anchor_rows(measurement)
    if anchors.empty:
        if args.strict:
            raise SystemExit("No valid direct anchors found for augmentation")
        base = measurement.copy()
        if "is_synthetic_regularization" not in base.columns:
            base["is_synthetic_regularization"] = False
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(args.output_csv, index=False)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "no_valid_anchors",
            "anchors": 0,
            "measurement_rows": int(len(base)),
            "output_csv": str(args.output_csv.resolve()),
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("No anchors available for augmentation; wrote direct rows only")
        return 0

    if int(args.max_anchors) > 0 and len(anchors) > int(args.max_anchors):
        perm = rng.permutation(len(anchors))
        anchors = anchors.iloc[perm[: int(args.max_anchors)]].reset_index(drop=True)

    nn, z, _, _ = _build_neighbor_index(single)
    row_to_pos = {int(r): int(i) for i, r in enumerate(single["row_index"].astype(int).tolist())}

    n_candidates = int(max(2, min(len(single), args.neighbor_candidates + 1)))
    n_pick = int(max(1, args.neighbors_per_anchor))
    tau = float(max(args.distance_tau, 1e-6))
    max_res_abs = float(abs(args.residual_max_abs_log10))

    synth_rows: List[Dict[str, object]] = []
    for _, a in anchors.iterrows():
        anchor_row = int(a["row_index"])
        if anchor_row not in row_to_pos:
            continue
        pos = row_to_pos[anchor_row]

        anchor_proxy_t1 = float(single.iloc[pos]["t1_estimate_us"])
        anchor_proxy_t2 = float(single.iloc[pos]["t2_estimate_us"])
        m_t1 = float(a.get("anchor_measured_t1_us", np.nan))
        m_t2 = float(a.get("anchor_measured_t2_us", np.nan))

        res_t1 = float("nan")
        res_t2 = float("nan")
        if np.isfinite(m_t1) and m_t1 > 0 and np.isfinite(anchor_proxy_t1) and anchor_proxy_t1 > 0:
            res_t1 = float(np.clip(np.log10(m_t1) - np.log10(anchor_proxy_t1), -max_res_abs, max_res_abs))
        if np.isfinite(m_t2) and m_t2 > 0 and np.isfinite(anchor_proxy_t2) and anchor_proxy_t2 > 0:
            res_t2 = float(np.clip(np.log10(m_t2) - np.log10(anchor_proxy_t2), -max_res_abs, max_res_abs))

        dist, idx = nn.kneighbors(z[pos : pos + 1], n_neighbors=n_candidates)
        d = dist[0]
        j = idx[0]

        picks = []
        for dd, jj in zip(d.tolist(), j.tolist()):
            if int(jj) == int(pos):
                continue
            picks.append((float(dd), int(jj)))
            if len(picks) >= n_pick:
                break

        for rank, (dd, jj) in enumerate(picks, start=1):
            blend = float(np.exp(-((dd / tau) ** 2)))
            if blend <= 1e-8:
                continue

            nrow = single.iloc[jj]
            proxy_t1 = float(nrow["t1_estimate_us"])
            proxy_t2 = float(nrow["t2_estimate_us"])

            synth_t1 = float("nan")
            synth_t2 = float("nan")
            if np.isfinite(res_t1) and np.isfinite(proxy_t1) and proxy_t1 > 0:
                synth_t1 = float(np.power(10.0, np.log10(proxy_t1) + blend * res_t1))
            if np.isfinite(res_t2) and np.isfinite(proxy_t2) and proxy_t2 > 0:
                synth_t2 = float(np.power(10.0, np.log10(proxy_t2) + blend * res_t2))

            if not (np.isfinite(synth_t1) or np.isfinite(synth_t2)):
                continue

            conf_base = float(max(1e-8, args.synthetic_confidence_scale * blend))
            conf = float(np.clip(conf_base * float(args.quality_weight_multiplier), args.synthetic_weight_min, args.synthetic_weight_max))

            synth_rows.append(
                {
                    "row_index": int(nrow["row_index"]),
                    "design_id": int(nrow["design_id"]),
                    "measured_t1_us": synth_t1,
                    "measured_t2_us": synth_t2,
                    "measured_freq_01_GHz": float(nrow["freq_01_GHz"]),
                    "measured_anharmonicity_GHz": float(nrow["anharmonicity_GHz"]),
                    "chip_id": "",
                    "cooldown_id": "",
                    "measurement_date_utc": "",
                    "source_file": "synthetic://anchor_conditioned_regularization",
                    "notes": "synthetic_anchor_regularization",
                    "source_name": "synthetic_anchor_regularization",
                    "source_record_id": f"synthetic_anchor_{anchor_row}_n{rank}",
                    "component_name": "",
                    "match_method": "anchor_neighbor_transfer",
                    "match_distance": float(dd),
                    "confidence_base": conf_base,
                    "source_confidence": 1.0,
                    "quality_weight_multiplier": float(args.quality_weight_multiplier),
                    "confidence_weight": conf,
                    "measurement_quality_tier": "synthetic_anchor_regularization",
                    "uncertainty_inflation_base": float(args.uncertainty_inflation_base),
                    "is_fitted_or_model_row": False,
                    "fit_t1_r2": np.nan,
                    "fit_t2_r2": np.nan,
                    "flux_mV": np.nan,
                    "is_synthetic_regularization": True,
                    "anchor_row_index": anchor_row,
                    "anchor_design_id": int(a.get("design_id", -1)),
                    "anchor_source_name": str(a.get("anchor_source_name", "") or ""),
                    "anchor_source_record_id": str(a.get("anchor_source_record_id", "") or ""),
                    "anchor_distance": float(dd),
                    "anchor_blend": float(blend),
                    "anchor_residual_t1_log10": res_t1,
                    "anchor_residual_t2_log10": res_t2,
                    "synthetic_generation": "palace_backed_local_transfer_v1",
                }
            )

    synth_df = pd.DataFrame(synth_rows)
    if not synth_df.empty:
        synth_df = synth_df.sort_values(["row_index", "anchor_blend", "anchor_distance"], ascending=[True, False, True]).copy()
        synth_df = synth_df.drop_duplicates(subset=["row_index", "anchor_row_index"], keep="first").copy()

        if int(args.max_synthetic_per_design) > 0:
            synth_df = (
                synth_df.groupby("row_index", group_keys=False, sort=False)
                .head(int(args.max_synthetic_per_design))
                .copy()
            )

    direct_df = measurement.loc[~measurement["is_synthetic_regularization"]].copy()
    direct_df["is_synthetic_regularization"] = False

    all_cols: List[str] = sorted(set(direct_df.columns.tolist()) | set(synth_df.columns.tolist()))
    out_direct = direct_df.reindex(columns=all_cols)
    out_synth = synth_df.reindex(columns=all_cols) if not synth_df.empty else pd.DataFrame(columns=all_cols)
    out = pd.concat([out_direct, out_synth], ignore_index=True)

    out = out.sort_values(["is_synthetic_regularization", "row_index", "confidence_weight"], ascending=[True, True, False]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    direct_rows = int((~out["is_synthetic_regularization"].astype(bool)).sum()) if "is_synthetic_regularization" in out.columns else int(len(out))
    synthetic_rows = int((out["is_synthetic_regularization"].astype(bool)).sum()) if "is_synthetic_regularization" in out.columns else 0

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_csv": str(args.single_csv.resolve()),
        "measurement_csv": str(args.measurement_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "params": {
            "neighbors_per_anchor": int(args.neighbors_per_anchor),
            "neighbor_candidates": int(args.neighbor_candidates),
            "distance_tau": float(args.distance_tau),
            "residual_max_abs_log10": float(max_res_abs),
            "max_anchors": int(args.max_anchors),
            "max_synthetic_per_design": int(args.max_synthetic_per_design),
            "synthetic_confidence_scale": float(args.synthetic_confidence_scale),
            "synthetic_weight_min": float(args.synthetic_weight_min),
            "synthetic_weight_max": float(args.synthetic_weight_max),
            "quality_weight_multiplier": float(args.quality_weight_multiplier),
            "uncertainty_inflation_base": float(args.uncertainty_inflation_base),
            "random_state": int(args.random_state),
        },
        "counts": {
            "anchors_used": int(len(anchors)),
            "rows_direct_out": int(direct_rows),
            "rows_synthetic_out": int(synthetic_rows),
            "rows_total_out": int(len(out)),
            "synthetic_unique_rows": int(out.loc[out["is_synthetic_regularization"] == True, "row_index"].nunique()) if ("is_synthetic_regularization" in out.columns and synthetic_rows > 0) else 0,
            "synthetic_unique_anchor_rows": int(out.loc[out["is_synthetic_regularization"] == True, "anchor_row_index"].nunique()) if ("anchor_row_index" in out.columns and synthetic_rows > 0) else 0,
        },
        "source_counts": {
            str(k): int(v)
            for k, v in out.groupby("source_name").size().to_dict().items()
            if "source_name" in out.columns
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Anchor-Conditioned Synthetic Regularization Complete ===")
    print(f"anchors_used={payload['counts']['anchors_used']} synthetic_rows={payload['counts']['rows_synthetic_out']} total_rows={payload['counts']['rows_total_out']}")
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
