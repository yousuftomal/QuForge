#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import h5py  # type: ignore
except Exception:
    h5py = None

BASE_CANONICAL_COLUMNS: Tuple[str, ...] = (
    "source_name",
    "source_record_id",
    "device_name",
    "component_name",
    "measured_freq_01_GHz",
    "measured_anharmonicity_GHz",
    "measured_t1_us",
    "measured_t2_us",
    "temperature_mK",
    "measurement_date_utc",
    "chip_id",
    "cooldown_id",
    "paper_link",
    "license",
    "raw_source_file",
    "quality_flags",
    "notes",
)

EXTRA_COLUMNS: Tuple[str, ...] = (
    "source_confidence",
    "fit_t1_r2",
    "fit_t2_r2",
    "flux_mV",
)

CANONICAL_COLUMNS: Tuple[str, ...] = (*BASE_CANONICAL_COLUMNS, *EXTRA_COLUMNS)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Augment canonical public dataset with large raw-source trace fits")
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=root / "Dataset" / "public_sources" / "bronze",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_large_raw_augmented.report.json",
    )
    parser.add_argument("--row-stride", type=int, default=50)
    parser.add_argument("--max-rows-fig4a", type=int, default=400)
    parser.add_argument("--min-r2", type=float, default=0.55)
    return parser.parse_args()


def parse_uncertain_float(v: Any) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m is None:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan


def fit_decay_t1_us(t_s: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t_s) & np.isfinite(y)
    if int(mask.sum()) < 12:
        return np.nan, np.nan, int(mask.sum())

    t_s = t_s[mask]
    y = y[mask]

    order = np.argsort(t_s)
    t_s = t_s[order]
    y = y[order]

    if t_s[-1] <= t_s[0]:
        return np.nan, np.nan, len(t_s)

    tail_n = max(4, int(0.15 * len(y)))
    baseline = float(np.median(y[-tail_n:]))

    amp = y - baseline
    head_n = max(3, int(0.10 * len(amp)))
    if float(np.median(amp[:head_n])) < 0:
        amp = -amp

    amp0 = float(np.max(amp)) if len(amp) > 0 else np.nan
    if (not np.isfinite(amp0)) or amp0 <= 0:
        return np.nan, np.nan, len(t_s)

    cutoff = max(amp0 * 0.05, 1e-9)
    m2 = amp > cutoff
    if int(np.sum(m2)) < 10:
        m2 = amp > 0
    if int(np.sum(m2)) < 10:
        return np.nan, np.nan, int(np.sum(m2))

    x = t_s[m2]
    z = np.log(amp[m2])

    try:
        slope, intercept = np.polyfit(x, z, deg=1)
    except Exception:
        return np.nan, np.nan, len(x)

    if slope >= -1e-12:
        return np.nan, np.nan, len(x)

    pred = slope * x + intercept
    sse = float(np.sum((z - pred) ** 2))
    sst = float(np.sum((z - np.mean(z)) ** 2) + 1e-12)
    r2 = 1.0 - sse / sst

    tau_s = -1.0 / slope
    t1_us = tau_s * 1e6

    if (not np.isfinite(t1_us)) or t1_us <= 0.001 or t1_us > 1e4:
        return np.nan, np.nan, len(x)

    return float(t1_us), float(r2), int(len(x))


def make_row(
    *,
    source_name: str,
    source_record_id: str,
    raw_source_file: str,
    t1_us: float = np.nan,
    t2_us: float = np.nan,
    fit_t1_r2: float = np.nan,
    notes: str = "",
    flags: Optional[List[str]] = None,
    source_confidence: float = 0.30,
) -> Dict[str, Any]:
    fl = list(flags or [])
    if not np.isfinite(t1_us) and not np.isfinite(t2_us):
        fl.append("missing_coherence")

    return {
        "source_name": source_name,
        "source_record_id": source_record_id,
        "device_name": source_name,
        "component_name": "qubit",
        "measured_freq_01_GHz": np.nan,
        "measured_anharmonicity_GHz": np.nan,
        "measured_t1_us": t1_us,
        "measured_t2_us": t2_us,
        "temperature_mK": np.nan,
        "measurement_date_utc": "",
        "chip_id": "",
        "cooldown_id": "",
        "paper_link": "",
        "license": "See upstream dataset licensing",
        "raw_source_file": raw_source_file,
        "quality_flags": ";".join(sorted(set(fl))),
        "notes": notes,
        "source_confidence": float(np.clip(source_confidence, 0.05, 1.0)),
        "fit_t1_r2": fit_t1_r2,
        "fit_t2_r2": np.nan,
        "flux_mV": np.nan,
    }


def extract_fig4a_tracefits(
    bronze_dir: Path,
    row_stride: int,
    max_rows: int,
    min_r2: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pex = bronze_dir / "data_gov_mds2_2516_jpg_3k_control__fig_4a_pex.csv"
    tlist = bronze_dir / "data_gov_mds2_2516_jpg_3k_control__fig_4a_tlist.csv"

    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "pex_exists": pex.exists(),
        "tlist_exists": tlist.exists(),
        "rows_scanned": 0,
        "rows_considered": 0,
        "rows_fit_ok": 0,
        "rows_added": 0,
    }

    if (not pex.exists()) or (not tlist.exists()):
        return rows, report

    report["raw_files"] = [str(pex.resolve()), str(tlist.resolve())]

    with pex.open("r", encoding="utf-8", errors="ignore") as fp, tlist.open("r", encoding="utf-8", errors="ignore") as ft:
        for ridx, (lp, lt) in enumerate(zip(fp, ft)):
            report["rows_scanned"] = ridx + 1

            if row_stride > 1 and (ridx % row_stride) != 0:
                continue
            report["rows_considered"] += 1

            p = np.fromstring(lp.strip(), sep=",")
            t = np.fromstring(lt.strip(), sep=",")
            if p.size < 20 or t.size < 20:
                continue

            n = int(min(p.size, t.size))
            p = p[:n]
            t = t[:n]

            t1_us, r2, used = fit_decay_t1_us(t_s=t, y=p)
            if (not np.isfinite(t1_us)) or (not np.isfinite(r2)):
                continue

            report["rows_fit_ok"] += 1

            if r2 < min_r2:
                continue

            flags = ["trace_fit", "missing_freq", "raw_matrix_fig4a"]
            if r2 < 0.75:
                flags.append("low_r2_t1")

            conf = float(np.clip(0.20 + 0.35 * r2, 0.20, 0.55))

            rows.append(
                make_row(
                    source_name="DataGov:NIST:mds2-2516_tracefit",
                    source_record_id=f"mds2-2516:fig4a:row{ridx}",
                    raw_source_file=f"{pex.resolve()};{tlist.resolve()}",
                    t1_us=t1_us,
                    fit_t1_r2=r2,
                    notes=f"tracefit_from_fig4a_matrix;row={ridx};points_used={used}",
                    flags=flags,
                    source_confidence=conf,
                )
            )

            if len(rows) >= max_rows:
                break

    report["rows_added"] = len(rows)
    return rows, report


def extract_zenodo_15364358_t1fit(bronze_dir: Path, min_r2: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = bronze_dir / "zenodo_15364358_low_latency_qec__figS4d_t1_time_fitting.csv"
    report: Dict[str, Any] = {"file_exists": path.exists(), "rows_added": 0}
    if not path.exists():
        return [], report

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        report["error"] = str(exc)
        return [], report

    if "delay_time" not in df.columns or "excited_state_population" not in df.columns:
        report["error"] = "expected columns not found"
        return [], report

    t_us = pd.to_numeric(df["delay_time"], errors="coerce").to_numpy(dtype=float)
    p = np.asarray([parse_uncertain_float(v) for v in df["excited_state_population"].to_numpy()], dtype=float)
    t_s = t_us * 1e-6

    t1_us, r2, used = fit_decay_t1_us(t_s=t_s, y=p)
    report["fit_t1_us"] = None if not np.isfinite(t1_us) else float(t1_us)
    report["fit_t1_r2"] = None if not np.isfinite(r2) else float(r2)
    report["points_used"] = int(used)

    if (not np.isfinite(t1_us)) or (not np.isfinite(r2)) or r2 < min_r2:
        return [], report

    conf = float(np.clip(0.35 + 0.45 * r2, 0.35, 0.80))
    row = make_row(
        source_name="Zenodo:15364358_tracefit",
        source_record_id="zenodo15364358:figS4d_t1_fit",
        raw_source_file=str(path.resolve()),
        t1_us=t1_us,
        fit_t1_r2=r2,
        notes=f"tracefit_from_figS4d_t1_time_fitting;points_used={used}",
        flags=["trace_fit", "missing_freq", "zenodo_15364358_figS4d"],
        source_confidence=conf,
    )
    report["rows_added"] = 1
    return [row], report


def inspect_h5_for_coherence_axes(bronze_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "h5py_available": h5py is not None,
        "file_exists": False,
        "t1_like_paths": [],
        "t2_like_paths": [],
    }
    path = bronze_dir / "zenodo_15364358_low_latency_qec__stability_8_without_resets_raw_data.h5"
    if not path.exists() or h5py is None:
        return out

    out["file_exists"] = True
    out["file"] = str(path.resolve())

    try:
        with h5py.File(path, "r") as h5:
            def walker(name: str, obj: Any) -> None:
                low = name.lower()
                if "t1" in low:
                    out["t1_like_paths"].append(name)
                if "t2" in low:
                    out["t2_like_paths"].append(name)

            h5.visititems(walker)
    except Exception as exc:
        out["error"] = str(exc)

    out["t1_like_paths"] = out["t1_like_paths"][:20]
    out["t2_like_paths"] = out["t2_like_paths"][:20]
    if len(out["t1_like_paths"]) == 0 and len(out["t2_like_paths"]) == 0:
        out["note"] = "no direct t1/t2 datasets in raw h5 structure; file kept for future custom decoder pipelines"
    return out


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = {
        "source_name",
        "source_record_id",
        "device_name",
        "component_name",
        "measurement_date_utc",
        "chip_id",
        "cooldown_id",
        "paper_link",
        "license",
        "raw_source_file",
        "quality_flags",
        "notes",
    }
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in text_cols else np.nan

    for col in [
        "measured_freq_01_GHz",
        "measured_anharmonicity_GHz",
        "measured_t1_us",
        "measured_t2_us",
        "temperature_mK",
        "source_confidence",
        "fit_t1_r2",
        "fit_t2_r2",
        "flux_mV",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.loc[:, list(CANONICAL_COLUMNS)]


def main() -> int:
    args = parse_args()

    if args.input_csv.exists():
        base = pd.read_csv(args.input_csv)
    else:
        base = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    base = ensure_columns(base)

    rows_fig4a, rep_fig4a = extract_fig4a_tracefits(
        bronze_dir=args.bronze_dir,
        row_stride=max(1, int(args.row_stride)),
        max_rows=max(1, int(args.max_rows_fig4a)),
        min_r2=float(args.min_r2),
    )
    rows_z1536, rep_z1536 = extract_zenodo_15364358_t1fit(
        bronze_dir=args.bronze_dir,
        min_r2=float(args.min_r2),
    )
    h5_rep = inspect_h5_for_coherence_axes(bronze_dir=args.bronze_dir)

    add_rows = rows_fig4a + rows_z1536
    add_df = pd.DataFrame(add_rows) if add_rows else pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    add_df = ensure_columns(add_df)

    merged = pd.concat([base, add_df], ignore_index=True)
    if "source_record_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["source_record_id"], keep="last")
    merged = merged.sort_values(["source_name", "source_record_id"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv.resolve()) if args.input_csv.exists() else str(args.input_csv),
        "output_csv": str(args.output_csv.resolve()),
        "bronze_dir": str(args.bronze_dir.resolve()),
        "params": {
            "row_stride": int(args.row_stride),
            "max_rows_fig4a": int(args.max_rows_fig4a),
            "min_r2": float(args.min_r2),
        },
        "rows_input": int(len(base)),
        "rows_added_total": int(len(add_df)),
        "rows_added_by_source": {
            "DataGov:NIST:mds2-2516_tracefit": int(len(rows_fig4a)),
            "Zenodo:15364358_tracefit": int(len(rows_z1536)),
        },
        "rows_output": int(len(merged)),
        "fig4a_report": rep_fig4a,
        "zenodo_15364358_report": rep_z1536,
        "h5_inspection": h5_rep,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Large Raw Source Augmentation Complete ===")
    print(f"rows_input={report['rows_input']} rows_added={report['rows_added_total']} rows_output={report['rows_output']}")
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

