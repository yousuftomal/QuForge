#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Conservatively map public measured rows to internal synthetic rows")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--public-csv", type=Path, default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.csv")
    parser.add_argument("--output-csv", type=Path, default=root / "Dataset" / "measurement_dataset_public_bootstrap.csv")
    parser.add_argument("--report-json", type=Path, default=root / "Dataset" / "measurement_dataset_public_bootstrap.report.json")

    parser.add_argument("--max-distance", type=float, default=1.2, help="Max normalized nearest-neighbor distance")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Min confidence score required to keep a match")
    parser.add_argument("--freq-only-penalty", type=float, default=0.70, help="Confidence penalty applied when only frequency is available")
    parser.add_argument("--measurement-weight-min", type=float, default=0.25)
    parser.add_argument("--measurement-weight-max", type=float, default=1.0)
    parser.add_argument(
        "--include-fitted-curves",
        action="store_true",
        help="Allow fitted/model curve rows during mapping (disabled by default)",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def robust_scale(x: np.ndarray) -> float:
    q25 = float(np.quantile(x, 0.25))
    q75 = float(np.quantile(x, 0.75))
    iqr = q75 - q25
    if iqr <= 1e-12:
        std = float(np.std(x))
        return std if std > 1e-12 else 1.0
    return iqr


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def is_fitted_row(row: pd.Series) -> bool:
    source = safe_text(row.get("source_name", "")).lower()
    notes = safe_text(row.get("notes", "")).lower()
    qflags = safe_text(row.get("quality_flags", "")).lower()

    if "tracefit" in source:
        return True

    tokens = ("model", "fit", "fitted", "simulated", "simulation")
    return any(tok in notes for tok in tokens) or any(tok in qflags for tok in tokens)


def main() -> int:
    args = parse_args()

    single_df = pd.read_csv(args.single_csv)
    ensure_columns(single_df, ["design_id", "freq_01_GHz", "anharmonicity_GHz"])
    single_df = single_df.reset_index().rename(columns={"index": "row_index"})

    public_df = pd.read_csv(args.public_csv)
    ensure_columns(
        public_df,
        [
            "source_name",
            "source_record_id",
            "measured_freq_01_GHz",
            "measured_anharmonicity_GHz",
            "measured_t1_us",
            "measured_t2_us",
            "measurement_date_utc",
            "raw_source_file",
            "notes",
        ],
    )

    public_df = public_df.copy()
    for col in ["measured_freq_01_GHz", "measured_anharmonicity_GHz", "measured_t1_us", "measured_t2_us"]:
        public_df[col] = pd.to_numeric(public_df[col], errors="coerce")

    if "source_confidence" in public_df.columns:
        public_df["source_confidence"] = pd.to_numeric(public_df["source_confidence"], errors="coerce")
    else:
        public_df["source_confidence"] = 1.0

    if not args.include_fitted_curves:
        fitted_mask = public_df.apply(is_fitted_row, axis=1)
        rejected_fitted = int(fitted_mask.sum())
        public_df = public_df.loc[~fitted_mask].copy()
    else:
        rejected_fitted = 0

    public_df = public_df[(public_df["measured_t1_us"].notna()) | (public_df["measured_t2_us"].notna())].copy()

    freq_internal = single_df["freq_01_GHz"].to_numpy(dtype=float)
    anh_internal = single_df["anharmonicity_GHz"].to_numpy(dtype=float)
    freq_scale = robust_scale(freq_internal)
    anh_scale = robust_scale(anh_internal)

    x2 = np.column_stack([freq_internal / freq_scale, anh_internal / anh_scale]).astype(float)
    x1 = (freq_internal.reshape(-1, 1) / freq_scale).astype(float)

    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn1 = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn2.fit(x2)
    nn1.fit(x1)

    matched_rows: List[Dict[str, object]] = []
    rejected_missing_freq = 0
    rejected_threshold = 0

    for _, row in public_df.iterrows():
        freq = float(row["measured_freq_01_GHz"]) if np.isfinite(row["measured_freq_01_GHz"]) else np.nan
        anh = float(row["measured_anharmonicity_GHz"]) if np.isfinite(row["measured_anharmonicity_GHz"]) else np.nan

        if not np.isfinite(freq):
            rejected_missing_freq += 1
            continue

        if np.isfinite(anh):
            q = np.array([[freq / freq_scale, anh / anh_scale]], dtype=float)
            dist, idx = nn2.kneighbors(q, n_neighbors=1)
            method = "freq+anh"
            norm_dist = float(dist[0, 0])
            confidence_base = float(np.exp(-norm_dist))
        else:
            q = np.array([[freq / freq_scale]], dtype=float)
            dist, idx = nn1.kneighbors(q, n_neighbors=1)
            method = "freq_only"
            norm_dist = float(dist[0, 0])
            confidence_base = float(np.exp(-norm_dist) * args.freq_only_penalty)

        src_conf = float(row.get("source_confidence", 1.0)) if np.isfinite(row.get("source_confidence", np.nan)) else 1.0
        src_conf = float(np.clip(src_conf, 0.05, 1.0))
        confidence = confidence_base * src_conf

        if norm_dist > args.max_distance or confidence < args.min_confidence:
            rejected_threshold += 1
            continue

        internal_i = int(idx[0, 0])
        mapped = single_df.iloc[internal_i]

        c_weight = float(np.clip(confidence, args.measurement_weight_min, args.measurement_weight_max))
        notes = safe_text(row.get("notes", ""))
        record_ref = safe_text(row.get("source_record_id", ""))
        notes = f"{notes} | public_record={record_ref}" if notes else f"public_record={record_ref}"

        matched_rows.append(
            {
                "row_index": int(mapped["row_index"]),
                "design_id": int(mapped["design_id"]),
                "measured_t1_us": row.get("measured_t1_us", np.nan),
                "measured_t2_us": row.get("measured_t2_us", np.nan),
                "measured_freq_01_GHz": row.get("measured_freq_01_GHz", np.nan),
                "measured_anharmonicity_GHz": row.get("measured_anharmonicity_GHz", np.nan),
                "chip_id": safe_text(row.get("chip_id", "")),
                "cooldown_id": safe_text(row.get("cooldown_id", "")),
                "measurement_date_utc": safe_text(row.get("measurement_date_utc", "")),
                "source_file": safe_text(row.get("raw_source_file", "")),
                "notes": notes,
                "source_name": safe_text(row.get("source_name", "")),
                "source_record_id": safe_text(row.get("source_record_id", "")),
                "component_name": safe_text(row.get("component_name", "")),
                "match_method": method,
                "match_distance": norm_dist,
                "confidence_base": confidence_base,
                "source_confidence": src_conf,
                "confidence_weight": c_weight,
                "fit_t1_r2": row.get("fit_t1_r2", np.nan),
                "fit_t2_r2": row.get("fit_t2_r2", np.nan),
                "flux_mV": row.get("flux_mV", np.nan),
            }
        )

    out_df = pd.DataFrame(matched_rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values(["row_index", "confidence_weight", "match_distance"], ascending=[True, False, True])
        out_df = out_df.drop_duplicates(subset=["row_index"], keep="first")
        out_df = out_df.sort_values("row_index").reset_index(drop=True)

    if args.strict and len(out_df) == 0:
        raise SystemExit("No mapped rows passed thresholds")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    source_counts: Dict[str, int] = {}
    method_counts: Dict[str, int] = {}
    if len(out_df) > 0:
        source_counts = {str(k): int(v) for k, v in out_df.groupby("source_name").size().to_dict().items()}
        method_counts = {str(k): int(v) for k, v in out_df.groupby("match_method").size().to_dict().items()}

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_csv": str(args.single_csv.resolve()),
        "public_csv": str(args.public_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "thresholds": {
            "max_distance": float(args.max_distance),
            "min_confidence": float(args.min_confidence),
            "freq_only_penalty": float(args.freq_only_penalty),
            "measurement_weight_min": float(args.measurement_weight_min),
            "measurement_weight_max": float(args.measurement_weight_max),
            "include_fitted_curves": bool(args.include_fitted_curves),
        },
        "counts": {
            "public_rows_input": int(len(public_df)),
            "mapped_rows_final": int(len(out_df)),
            "rejected_missing_freq": int(rejected_missing_freq),
            "rejected_threshold": int(rejected_threshold),
            "rejected_fitted_curve_rows": int(rejected_fitted),
            "unique_designs_mapped": int(out_df["design_id"].nunique()) if len(out_df) > 0 else 0,
        },
        "confidence_summary": {
            "min": float(out_df["confidence_weight"].min()) if len(out_df) > 0 else None,
            "median": float(out_df["confidence_weight"].median()) if len(out_df) > 0 else None,
            "max": float(out_df["confidence_weight"].max()) if len(out_df) > 0 else None,
            "source_confidence_median": float(out_df["source_confidence"].median()) if len(out_df) > 0 else None,
        },
        "source_counts": source_counts,
        "method_counts": method_counts,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Public->Internal Mapping Complete ===")
    print(
        f"mapped_rows_final={report['counts']['mapped_rows_final']} "
        f"rejected_missing_freq={report['counts']['rejected_missing_freq']} "
        f"rejected_threshold={report['counts']['rejected_threshold']} "
        f"rejected_fitted={report['counts']['rejected_fitted_curve_rows']}"
    )
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())