#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def detect_col(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Column not found: {explicit}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build and QA Dataset/measurement_dataset.csv from raw lab measurements")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--raw-csv", type=Path, required=True, help="Raw lab measurement export CSV")
    parser.add_argument("--output-csv", type=Path, default=root / "Dataset" / "measurement_dataset.csv")
    parser.add_argument("--report-json", type=Path, default=root / "Dataset" / "measurement_dataset.report.json")

    parser.add_argument("--row-index-col", type=str, default=None)
    parser.add_argument("--design-id-col", type=str, default=None)
    parser.add_argument("--t1-col", type=str, default=None)
    parser.add_argument("--t2-col", type=str, default=None)
    parser.add_argument("--freq-col", type=str, default=None)
    parser.add_argument("--anh-col", type=str, default=None)
    parser.add_argument("--chip-col", type=str, default=None)
    parser.add_argument("--cooldown-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--notes-col", type=str, default=None)

    parser.add_argument("--dedup", choices=("latest", "mean"), default="latest")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base = pd.read_csv(args.single_csv)
    if "design_id" not in base.columns:
        raise SystemExit("single-csv missing design_id")

    base = base.reset_index().rename(columns={"index": "row_index"})
    map_row_to_design = dict(zip(base["row_index"].astype(int), base["design_id"].astype(int)))
    map_design_to_row = dict(zip(base["design_id"].astype(int), base["row_index"].astype(int)))

    raw = pd.read_csv(args.raw_csv)
    if raw.empty:
        raise SystemExit("raw-csv is empty")

    row_col = detect_col(raw, args.row_index_col, ["row_index", "rowid", "row_id", "dataset_row_index"])
    design_col = detect_col(raw, args.design_id_col, ["design_id", "designid", "design"])
    if row_col is None and design_col is None:
        raise SystemExit("Need row index or design id column in raw-csv")

    t1_col = detect_col(raw, args.t1_col, ["measured_t1_us", "t1_us", "T1_us", "T1", "t1"])
    t2_col = detect_col(raw, args.t2_col, ["measured_t2_us", "t2_us", "T2_us", "T2", "t2"])
    freq_col = detect_col(raw, args.freq_col, ["measured_freq_01_GHz", "freq_01_GHz", "f01_GHz", "f01"])
    anh_col = detect_col(raw, args.anh_col, ["measured_anharmonicity_GHz", "anharmonicity_GHz", "alpha_GHz", "anh"])
    chip_col = detect_col(raw, args.chip_col, ["chip_id", "chip", "wafer_chip_id"])
    cooldown_col = detect_col(raw, args.cooldown_col, ["cooldown_id", "cooldown", "run_id"])
    date_col = detect_col(raw, args.date_col, ["measurement_date_utc", "measurement_date", "date", "timestamp"])
    notes_col = detect_col(raw, args.notes_col, ["notes", "comment", "remarks"])

    if t1_col is None and t2_col is None:
        raise SystemExit("Need at least one coherence column (T1 or T2)")

    work = pd.DataFrame(index=raw.index)

    if row_col is not None:
        work["row_index"] = parse_num(raw[row_col])
    else:
        work["row_index"] = np.nan

    if design_col is not None:
        work["design_id"] = parse_num(raw[design_col])
    else:
        work["design_id"] = np.nan

    if t1_col is not None:
        work["measured_t1_us"] = parse_num(raw[t1_col])
    else:
        work["measured_t1_us"] = np.nan

    if t2_col is not None:
        work["measured_t2_us"] = parse_num(raw[t2_col])
    else:
        work["measured_t2_us"] = np.nan

    work["measured_freq_01_GHz"] = parse_num(raw[freq_col]) if freq_col is not None else np.nan
    work["measured_anharmonicity_GHz"] = parse_num(raw[anh_col]) if anh_col is not None else np.nan

    work["chip_id"] = raw[chip_col].astype(str) if chip_col is not None else ""
    work["cooldown_id"] = raw[cooldown_col].astype(str) if cooldown_col is not None else ""
    work["notes"] = raw[notes_col].astype(str) if notes_col is not None else ""
    work["measurement_date_utc"] = raw[date_col].astype(str) if date_col is not None else ""

    # Fill missing IDs from each other where possible.
    missing_row_before = int(work["row_index"].isna().sum())
    missing_design_before = int(work["design_id"].isna().sum())

    if work["row_index"].isna().any() and work["design_id"].notna().any():
        work.loc[work["row_index"].isna(), "row_index"] = work.loc[work["row_index"].isna(), "design_id"].map(map_design_to_row)

    if work["design_id"].isna().any() and work["row_index"].notna().any():
        work.loc[work["design_id"].isna(), "design_id"] = work.loc[work["design_id"].isna(), "row_index"].map(map_row_to_design)

    work["row_index"] = parse_num(work["row_index"]).astype("Int64")
    work["design_id"] = parse_num(work["design_id"]).astype("Int64")

    # Keep rows that have at least one measured coherence value.
    work = work[(work["measured_t1_us"].notna()) | (work["measured_t2_us"].notna())].copy()

    # Basic physical sanity checks.
    invalid_t1 = int(((work["measured_t1_us"].notna()) & (work["measured_t1_us"] <= 0)).sum())
    invalid_t2 = int(((work["measured_t2_us"].notna()) & (work["measured_t2_us"] <= 0)).sum())
    work.loc[work["measured_t1_us"] <= 0, "measured_t1_us"] = np.nan
    work.loc[work["measured_t2_us"] <= 0, "measured_t2_us"] = np.nan

    # Drop rows still without usable coherence values.
    work = work[(work["measured_t1_us"].notna()) | (work["measured_t2_us"].notna())].copy()

    # IDs must resolve to dataset rows.
    base_rows = set(map_row_to_design.keys())
    base_designs = set(map_design_to_row.keys())

    valid_row = work["row_index"].notna() & work["row_index"].astype(int).isin(base_rows)
    valid_design = work["design_id"].notna() & work["design_id"].astype(int).isin(base_designs)
    valid_any = valid_row | valid_design

    dropped_unmapped = int((~valid_any).sum())
    work = work[valid_any].copy()

    # Canonicalize both IDs and check consistency.
    work.loc[work["row_index"].isna(), "row_index"] = work.loc[work["row_index"].isna(), "design_id"].astype(int).map(map_design_to_row)
    work.loc[work["design_id"].isna(), "design_id"] = work.loc[work["design_id"].isna(), "row_index"].astype(int).map(map_row_to_design)

    inferred_design = work["row_index"].astype(int).map(map_row_to_design)
    mismatch_mask = inferred_design.astype(int) != work["design_id"].astype(int)
    mismatched_ids = int(mismatch_mask.sum())
    if mismatched_ids > 0:
        if args.strict:
            raise SystemExit(f"Found {mismatched_ids} row_index/design_id mismatches")
        work = work[~mismatch_mask].copy()

    if date_col is not None:
        parsed_date = pd.to_datetime(work["measurement_date_utc"], utc=True, errors="coerce")
        work["_date"] = parsed_date
    else:
        work["_date"] = pd.NaT

    pre_dedup_rows = int(len(work))

    if args.dedup == "latest" and work["_date"].notna().any():
        work = work.sort_values(["row_index", "_date"]).groupby("row_index", as_index=False).tail(1)
    else:
        agg_cols = {
            "design_id": "first",
            "measured_t1_us": "mean",
            "measured_t2_us": "mean",
            "measured_freq_01_GHz": "mean",
            "measured_anharmonicity_GHz": "mean",
            "chip_id": "last",
            "cooldown_id": "last",
            "measurement_date_utc": "last",
            "notes": "last",
        }
        work = work.groupby("row_index", as_index=False).agg(agg_cols)

    post_dedup_rows = int(len(work))

    work["row_index"] = work["row_index"].astype(int)
    work["design_id"] = work["design_id"].astype(int)
    work["source_file"] = str(args.raw_csv)

    out_cols = [
        "row_index",
        "design_id",
        "measured_t1_us",
        "measured_t2_us",
        "measured_freq_01_GHz",
        "measured_anharmonicity_GHz",
        "chip_id",
        "cooldown_id",
        "measurement_date_utc",
        "source_file",
        "notes",
    ]
    out = work[out_cols].sort_values("row_index").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    report: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_csv": str(args.single_csv.resolve()),
        "raw_csv": str(args.raw_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows_raw": int(len(raw)),
        "rows_after_nonempty_coherence": int(pre_dedup_rows),
        "rows_after_dedup": int(post_dedup_rows),
        "rows_final": int(len(out)),
        "dedup_mode": args.dedup,
        "id_columns_detected": {
            "row_index_col": row_col,
            "design_id_col": design_col,
        },
        "measurement_columns_detected": {
            "t1_col": t1_col,
            "t2_col": t2_col,
            "freq_col": freq_col,
            "anh_col": anh_col,
            "date_col": date_col,
        },
        "sanity": {
            "invalid_t1_nonpositive": invalid_t1,
            "invalid_t2_nonpositive": invalid_t2,
            "dropped_unmapped_rows": dropped_unmapped,
            "id_mismatch_rows": mismatched_ids,
            "missing_row_index_before_fill": missing_row_before,
            "missing_design_id_before_fill": missing_design_before,
            "final_rows_with_t1": int(out["measured_t1_us"].notna().sum()),
            "final_rows_with_t2": int(out["measured_t2_us"].notna().sum()),
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Measurement Dataset Build Complete ===")
    print(f"rows_final={len(out)}")
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
