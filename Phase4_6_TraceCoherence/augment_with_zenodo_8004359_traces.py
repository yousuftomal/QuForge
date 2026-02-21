#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    "trace_rows_t1",
    "trace_rows_t2",
    "flux_mV",
)




def safe_nanptp(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float("nan")
    try:
        return float(np.nanmax(arr) - np.nanmin(arr))
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Augment canonical public dataset with trace-fitted coherence from Zenodo 8004359")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.csv",
    )
    parser.add_argument(
        "--zenodo-zip",
        type=Path,
        default=root / "Dataset" / "public_sources" / "bronze" / "zenodo_8004359.zip",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical_augmented.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical_augmented.report.json",
    )
    parser.add_argument("--min-r2", type=float, default=0.5)
    parser.add_argument("--min-trace-points", type=int, default=16)
    return parser.parse_args()


def parse_numeric_line(line: str) -> List[float]:
    vals: List[float] = []
    for tok in line.replace(",", "\t").split("\t"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def extract_trace_blocks(text: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    lines = text.splitlines()
    t_ns: List[float] = []
    blocks: List[np.ndarray] = []

    for i, line in enumerate(lines):
        if "M4i_sweep" in line:
            for j in range(i + 1, min(i + 8, len(lines))):
                vals = parse_numeric_line(lines[j])
                if len(vals) > 10:
                    t_ns = vals
                    break
            if t_ns:
                break

    i = 0
    while i < len(lines):
        if "Hello Matilda" in lines[i]:
            i += 1
            rows: List[List[float]] = []
            while i < len(lines):
                if lines[i].strip().startswith("%%%%%%%%%%"):
                    break
                vals = parse_numeric_line(lines[i])
                if len(vals) > 10:
                    rows.append(vals)
                i += 1
            if rows:
                lens = [len(r) for r in rows if len(r) > 0]
                if lens:
                    target = int(pd.Series(lens).mode().iloc[0])
                    clean_rows = [r[:target] for r in rows if len(r) >= target]
                    if clean_rows:
                        arr = np.asarray(clean_rows, dtype=float)
                        blocks.append(arr)
        i += 1

    return np.asarray(t_ns, dtype=float), blocks


def fit_decay_tau_us(t_ns: np.ndarray, block: np.ndarray) -> Tuple[float, float, int, float]:
    if t_ns.size < 8 or block.size == 0:
        return np.nan, np.nan, 0, np.nan

    y = np.nanmedian(block, axis=0)
    n = min(len(t_ns), len(y))
    if n < 8:
        return np.nan, np.nan, 0, np.nan

    t_us = np.asarray(t_ns[:n], dtype=float) / 1000.0
    y = np.asarray(y[:n], dtype=float)

    tail_start = max(1, int(0.85 * n))
    baseline = float(np.nanmedian(y[tail_start:]))
    amp = y - baseline

    head_end = max(3, int(0.10 * n))
    if float(np.nanmedian(amp[:head_end])) < 0:
        amp = -amp

    amp0 = float(np.nanmax(amp)) if np.isfinite(np.nanmax(amp)) else np.nan
    if not np.isfinite(amp0) or amp0 <= 0:
        return np.nan, np.nan, 0, safe_nanptp(y)

    threshold = max(amp0 * 0.08, 1e-12)
    mask = amp > threshold
    if int(np.sum(mask)) < 8:
        mask = amp > 0
    if int(np.sum(mask)) < 8:
        return np.nan, np.nan, int(np.sum(mask)), safe_nanptp(y)

    x = t_us[mask]
    z = np.log(amp[mask])

    try:
        slope, intercept = np.polyfit(x, z, deg=1)
    except Exception:
        return np.nan, np.nan, int(np.sum(mask)), safe_nanptp(y)

    if slope >= -1e-12:
        return np.nan, np.nan, int(np.sum(mask)), safe_nanptp(y)

    pred = slope * x + intercept
    sse = float(np.sum((z - pred) ** 2))
    sst = float(np.sum((z - np.mean(z)) ** 2) + 1e-12)
    r2 = 1.0 - sse / sst

    tau_us = -1.0 / slope
    if not np.isfinite(tau_us) or tau_us <= 0 or tau_us > 1e8:
        return np.nan, np.nan, int(np.sum(mask)), safe_nanptp(y)

    return float(tau_us), float(r2), int(np.sum(mask)), safe_nanptp(y)


def pick_best_block_fit(t_ns: np.ndarray, blocks: List[np.ndarray]) -> Tuple[float, float, int, float]:
    if t_ns.size == 0 or not blocks:
        return np.nan, np.nan, 0, np.nan

    best = (np.nan, np.nan, 0, np.nan)
    best_score = -np.inf

    for block in blocks:
        tau, r2, used_points, span = fit_decay_tau_us(t_ns, block)
        if not np.isfinite(tau) or not np.isfinite(r2):
            continue
        score = r2 + 0.05 * np.log10(max(used_points, 1))
        if score > best_score:
            best_score = score
            best = (tau, r2, used_points, span)

    return best


def parse_flux_token(path: str) -> Optional[Tuple[str, str, str]]:
    m = re.search(r"/Device\s+([A-Z])/T([12])\s+Vs\s+Flux/T[12]_flu?x?([N]?\d+)(?:[_-]\d+)?\.txt$", path)
    if m is None:
        return None
    device = m.group(1)
    t_kind = "t1" if m.group(2) == "1" else "t2"
    flux_token = m.group(3)
    return device, t_kind, flux_token


def flux_token_to_mv(token: str) -> Optional[float]:
    if not token:
        return None
    if token.startswith("N") and len(token) > 1:
        try:
            return -float(token[1:])
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_spec_file(text: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lines = text.splitlines()
    mode = ""
    voltage = None
    freq = None
    rows: List[List[float]] = []

    for line in lines:
        s = line.strip().lower()
        if s.startswith("%%%%%%%%%%"):
            if "voltage_source" in s:
                mode = "v"
                continue
            if "r_freq_source" in s:
                mode = "f"
                continue
            if "hello matilda" in s:
                mode = "d"
                continue

        vals = parse_numeric_line(line)
        if not vals:
            continue

        if mode == "v" and voltage is None and len(vals) > 20:
            voltage = np.asarray(vals, dtype=float)
            continue
        if mode == "f" and freq is None and len(vals) > 20:
            freq = np.asarray(vals, dtype=float)
            continue
        if mode == "d" and len(vals) > 20:
            rows.append(vals)

    if voltage is None or freq is None or not rows:
        return None

    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] != len(freq):
        return None

    if arr.shape[0] == 2 * len(voltage):
        arr = 0.5 * (arr[: len(voltage)] + arr[len(voltage) :])
    elif arr.shape[0] > len(voltage):
        arr = arr[: len(voltage)]

    if arr.shape[0] != len(voltage):
        return None

    return voltage, freq, arr


def build_device_b_freq_map(zf: zipfile.ZipFile) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    spec_names = [n for n in zf.namelist() if "Fig5/Fig5c/specge" in n and n.lower().endswith(".txt")]
    if not spec_names:
        return None

    v_ref = None
    f_ref = None
    mats: List[np.ndarray] = []

    for name in sorted(spec_names):
        parsed = parse_spec_file(zf.read(name).decode("utf-8", errors="ignore"))
        if parsed is None:
            continue
        v, f, m = parsed
        if v_ref is None:
            v_ref = v
            f_ref = f
        if len(v) != len(v_ref) or len(f) != len(f_ref):
            continue

        m_norm = m.copy()
        m_norm = (m_norm - np.median(m_norm, axis=1, keepdims=True)) / (np.std(m_norm, axis=1, keepdims=True) + 1e-12)
        mats.append(m_norm)

    if v_ref is None or f_ref is None or not mats:
        return None

    m_avg = np.mean(np.stack(mats, axis=0), axis=0)
    idx = np.argmin(m_avg, axis=1)
    f_est = f_ref[idx]
    return v_ref, f_est


def extract_trace_rows(
    zip_path: Path,
    min_r2: float,
    min_trace_points: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not zip_path.exists():
        return pd.DataFrame(), {"error": f"zip not found: {zip_path}"}

    grouped: Dict[Tuple[str, str], Dict[str, List[Tuple[float, float, int, float]]]] = {}

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        freq_map = build_device_b_freq_map(zf)
        device_b_v = freq_map[0] if freq_map is not None else None
        device_b_f = freq_map[1] if freq_map is not None else None

        trace_names = [
            n
            for n in zf.namelist()
            if "/Fig6/" in n and "/T1 Vs Flux/" in n or "/Fig6/" in n and "/T2 Vs Flux/" in n
        ]
        trace_names = [n for n in trace_names if n.lower().endswith(".txt")]

        parsed_files = 0
        accepted_fits = 0

        for name in sorted(trace_names):
            meta = parse_flux_token(name)
            if meta is None:
                continue
            device, t_kind, flux_token = meta

            text = zf.read(name).decode("utf-8", errors="ignore")
            t_ns, blocks = extract_trace_blocks(text)
            tau_us, r2, used_points, span = pick_best_block_fit(t_ns, blocks)
            parsed_files += 1

            if (not np.isfinite(tau_us)) or (not np.isfinite(r2)):
                continue
            if r2 < min_r2 or used_points < min_trace_points:
                continue

            key = (device, flux_token)
            if key not in grouped:
                grouped[key] = {"t1": [], "t2": []}
            grouped[key][t_kind].append((tau_us, r2, used_points, span))
            accepted_fits += 1

    rows: List[Dict[str, object]] = []
    with_freq = 0

    for (device, flux_token), vals in sorted(grouped.items()):
        t1_list = vals.get("t1", [])
        t2_list = vals.get("t2", [])

        if not t1_list and not t2_list:
            continue

        t1_us = float(np.median([x[0] for x in t1_list])) if t1_list else np.nan
        t2_us = float(np.median([x[0] for x in t2_list])) if t2_list else np.nan

        fit_t1_r2 = float(np.median([x[1] for x in t1_list])) if t1_list else np.nan
        fit_t2_r2 = float(np.median([x[1] for x in t2_list])) if t2_list else np.nan

        flux_mv = flux_token_to_mv(flux_token)
        freq_ghz = np.nan
        freq_flag = "missing_freq"

        if device == "B" and flux_mv is not None:
            # Device-B spectroscopy map is measured in Fig5c (~voltage_source mV -> f01 GHz).
            if device_b_v is not None and device_b_f is not None:
                if float(np.nanmin(device_b_v)) <= flux_mv <= float(np.nanmax(device_b_v)):
                    freq_ghz = float(np.interp(flux_mv, device_b_v, device_b_f))
                    freq_flag = "freq_inferred_from_specge"
                    with_freq += 1
                else:
                    freq_flag = "freq_out_of_spec_range"

        r2_values = [v for v in [fit_t1_r2, fit_t2_r2] if np.isfinite(v)]
        fit_conf = float(np.mean(r2_values)) if r2_values else 0.0
        has_freq = np.isfinite(freq_ghz)
        source_conf = 0.2 + 0.5 * fit_conf + (0.2 if has_freq else 0.0)
        source_conf = float(np.clip(source_conf, 0.20, 0.90))

        flags = ["trace_fit", freq_flag]
        if np.isfinite(fit_t1_r2) and fit_t1_r2 < 0.70:
            flags.append("low_r2_t1")
        if np.isfinite(fit_t2_r2) and fit_t2_r2 < 0.70:
            flags.append("low_r2_t2")

        row = {
            "source_name": "Zenodo:8004359_tracefit",
            "source_record_id": f"zenodo8004359:{device}:flux{flux_token}",
            "device_name": f"IST_Device_{device}",
            "component_name": "qubit",
            "measured_freq_01_GHz": freq_ghz,
            "measured_anharmonicity_GHz": np.nan,
            "measured_t1_us": t1_us,
            "measured_t2_us": t2_us,
            "temperature_mK": np.nan,
            "measurement_date_utc": "",
            "chip_id": f"IST_Device_{device}",
            "cooldown_id": f"flux_{flux_token}",
            "paper_link": "https://doi.org/10.5281/zenodo.8004359",
            "license": "See Zenodo record licensing",
            "raw_source_file": str(zip_path.resolve()),
            "quality_flags": ";".join(sorted(set(flags))),
            "notes": f"trace_fit_from_fig6;device={device};flux_token={flux_token}",
            "source_confidence": source_conf,
            "fit_t1_r2": fit_t1_r2,
            "fit_t2_r2": fit_t2_r2,
            "trace_rows_t1": int(len(t1_list)),
            "trace_rows_t2": int(len(t2_list)),
            "flux_mV": flux_mv,
        }
        rows.append(row)

    report = {
        "parsed_trace_files": int(parsed_files if 'parsed_files' in locals() else 0),
        "accepted_trace_fits": int(accepted_fits if 'accepted_fits' in locals() else 0),
        "rows_grouped": int(len(rows)),
        "rows_with_freq": int(with_freq),
        "rows_without_freq": int(max(len(rows) - with_freq, 0)),
        "source_name": "Zenodo:8004359_tracefit",
    }

    return pd.DataFrame(rows), report


def main() -> int:
    args = parse_args()

    if args.input_csv.exists():
        base = pd.read_csv(args.input_csv)
    else:
        base = pd.DataFrame(columns=list(BASE_CANONICAL_COLUMNS))

    extracted_df, extract_report = extract_trace_rows(
        zip_path=args.zenodo_zip,
        min_r2=args.min_r2,
        min_trace_points=args.min_trace_points,
    )

    for col in [*BASE_CANONICAL_COLUMNS, *EXTRA_COLUMNS]:
        if col not in base.columns:
            base[col] = np.nan if col not in {
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
            } else ""

    if not extracted_df.empty:
        for col in [*BASE_CANONICAL_COLUMNS, *EXTRA_COLUMNS]:
            if col not in extracted_df.columns:
                extracted_df[col] = np.nan if col not in {
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
                } else ""

        merged = pd.concat([base, extracted_df], ignore_index=True)
    else:
        merged = base.copy()

    if "source_record_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["source_record_id"], keep="last")

    merged = merged.sort_values(["source_name", "source_record_id"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv.resolve()) if args.input_csv.exists() else str(args.input_csv),
        "zenodo_zip": str(args.zenodo_zip.resolve()) if args.zenodo_zip.exists() else str(args.zenodo_zip),
        "output_csv": str(args.output_csv.resolve()),
        "min_r2": float(args.min_r2),
        "min_trace_points": int(args.min_trace_points),
        "rows_input": int(len(base)),
        "rows_extracted": int(len(extracted_df)),
        "rows_output": int(len(merged)),
        "extraction": extract_report,
        "rows_output_with_t1": int(pd.to_numeric(merged.get("measured_t1_us"), errors="coerce").notna().sum()) if "measured_t1_us" in merged else 0,
        "rows_output_with_t2": int(pd.to_numeric(merged.get("measured_t2_us"), errors="coerce").notna().sum()) if "measured_t2_us" in merged else 0,
        "rows_output_with_freq": int(pd.to_numeric(merged.get("measured_freq_01_GHz"), errors="coerce").notna().sum()) if "measured_freq_01_GHz" in merged else 0,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Phase 4.6 Trace Augmentation Complete ===")
    print(f"rows_extracted={report['rows_extracted']} rows_output={report['rows_output']}")
    print(f"rows_output_with_freq={report['rows_output_with_freq']}")
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())