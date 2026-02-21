#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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
    "flux_mV",
)

CANONICAL_COLUMNS: Tuple[str, ...] = (*BASE_CANONICAL_COLUMNS, *EXTRA_COLUMNS)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build canonical public measurement dataset from bronze sources")
    parser.add_argument("--bronze-dir", type=Path, default=root / "Dataset" / "public_sources" / "bronze")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=root / "Dataset" / "public_sources" / "silver" / "public_measurements_canonical.report.json",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--include-fitted-curves",
        action="store_true",
        help="Include fitted/model curve columns (disabled by default for stricter measured-data extraction)",
    )
    return parser.parse_args()


def to_float(value: Any) -> Tuple[float, Optional[str]]:
    if value is None:
        return np.nan, None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v, None if np.isfinite(v) else None
    if isinstance(value, list):
        vals: List[float] = []
        flags: List[str] = []
        for item in value:
            vi, fi = to_float(item)
            if np.isfinite(vi):
                vals.append(float(vi))
            if fi is not None:
                flags.append(fi)
        if not vals:
            return np.nan, (";".join(sorted(set(flags))) if flags else None)
        return float(np.median(vals)), (";".join(sorted(set(flags))) if flags else None)
    s = str(value).strip()
    if not s:
        return np.nan, None

    flag: Optional[str] = None
    if "<" in s:
        flag = "lt_bound"
    elif ">" in s:
        flag = "gt_bound"

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m is None:
        return np.nan, flag
    try:
        return float(m.group(0)), flag
    except ValueError:
        return np.nan, flag


def normalize_frequency_ghz(value: float, key: str) -> float:
    if not np.isfinite(value):
        return np.nan
    k = key.lower()
    v = float(value)
    if "mhz" in k:
        v /= 1e3
    elif "khz" in k:
        v /= 1e6
    elif "hz" in k and "ghz" not in k:
        v /= 1e9
    else:
        if abs(v) > 1e6:
            v /= 1e9
        elif abs(v) > 1e3:
            v /= 1e6
        elif abs(v) > 100:
            v /= 1e3
    if v <= 0 or v > 50:
        return np.nan
    return v


def normalize_anh_ghz(value: float, key: str) -> float:
    if not np.isfinite(value):
        return np.nan
    k = key.lower()
    v = float(value)
    if "mhz" in k:
        v /= 1e3
    elif "khz" in k:
        v /= 1e6
    elif "hz" in k and "ghz" not in k:
        v /= 1e9
    else:
        if abs(v) > 1e6:
            v /= 1e9
        elif abs(v) > 1e3:
            v /= 1e6
        elif abs(v) > 5:
            v /= 1e3
    if not np.isfinite(v) or abs(v) > 5:
        return np.nan
    return v


def normalize_time_us(value: float, key: str) -> float:
    if not np.isfinite(value):
        return np.nan
    k = key.lower()
    v = float(value)
    if "ns" in k:
        v /= 1e3
    elif re.search(r"(?<!u)ms", k):
        v *= 1e3
    elif re.search(r"(^|[^a-z])s($|[^a-z])", k) and "us" not in k:
        v *= 1e6
    if v <= 0 or v > 1e8:
        return np.nan
    return v


def normalize_temp_mk(value: float, key: str) -> float:
    if not np.isfinite(value):
        return np.nan
    k = key.lower()
    v = float(value)
    if "mk" in k:
        pass
    elif "kelvin" in k or re.search(r"(^|[^a-z])k($|[^a-z])", k):
        v *= 1e3
    if v <= 0 or v > 1e6:
        return np.nan
    return v


def iter_dict_nodes(obj: Any, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    if isinstance(obj, dict):
        yield path, obj
        for key, value in obj.items():
            yield from iter_dict_nodes(value, path + (str(key),))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            yield from iter_dict_nodes(item, path + (f"[{idx}]",))


def extract_component_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    freq = np.nan
    anh = np.nan
    t1 = np.nan
    t2 = np.nan
    temp = np.nan
    flags: List[str] = []

    for key, raw_val in d.items():
        key_l = str(key).lower()
        val, flag = to_float(raw_val)
        if flag is not None:
            flags.append(flag)
        if not np.isfinite(val):
            continue

        if (
            "f_01" in key_l
            or "f01" in key_l
            or "qubit_frequency" in key_l
            or "qubit freq" in key_l
            or ("frequency" in key_l and "resonator" not in key_l and "cavity" not in key_l)
            or ("omega" in key_l and "resonator" not in key_l and "cavity" not in key_l)
        ):
            cand = normalize_frequency_ghz(val, key_l)
            if np.isfinite(cand) and not np.isfinite(freq):
                freq = cand

        if "anh" in key_l or "alpha" in key_l:
            cand = normalize_anh_ghz(val, key_l)
            if np.isfinite(cand) and not np.isfinite(anh):
                anh = cand

        if "t1" in key_l:
            cand = normalize_time_us(val, key_l)
            if np.isfinite(cand) and not np.isfinite(t1):
                t1 = cand

        if "t2" in key_l:
            cand = normalize_time_us(val, key_l)
            if np.isfinite(cand) and not np.isfinite(t2):
                t2 = cand

        if "temp" in key_l or "temperature" in key_l:
            cand = normalize_temp_mk(val, key_l)
            if np.isfinite(cand) and not np.isfinite(temp):
                temp = cand

    return {
        "measured_freq_01_GHz": freq,
        "measured_anharmonicity_GHz": anh,
        "measured_t1_us": t1,
        "measured_t2_us": t2,
        "temperature_mK": temp,
        "flags": sorted(set(flags)),
    }


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def find_cols(columns: Sequence[str], candidates: Sequence[str], excludes: Sequence[str] = ()) -> List[str]:
    out: List[str] = []
    for col in columns:
        low = str(col).lower()
        if any(exc in low for exc in excludes):
            continue
        if any(cand in low for cand in candidates):
            out.append(col)
    return out


def first_col(columns: Sequence[str], candidates: Sequence[str], excludes: Sequence[str] = ()) -> Optional[str]:
    cols = find_cols(columns, candidates, excludes)
    return cols[0] if cols else None


def component_from_column(col_name: str) -> str:
    low = str(col_name).lower()
    m = re.search(r"(q\d+|qb\d+|qubit\d+|device[_ ]?[a-z0-9]+)", low)
    if m:
        return m.group(1).replace(" ", "")
    return "main"


def human_source_name(raw: str) -> str:
    if raw.startswith("squadds"):
        return "SQuADDS"
    if raw.startswith("zenodo_"):
        m = re.search(r"zenodo_(\d+)", raw)
        if m:
            return f"Zenodo:{m.group(1)}"
    if raw.startswith("data_gov_"):
        m = re.search(r"mds2_(\d+)", raw)
        if m:
            return f"DataGov:NIST:mds2-{m.group(1)}"
        return "DataGov:NIST"
    return raw


def source_parts_from_filename(path: Path) -> Tuple[str, str]:
    name = path.name
    if "__" in name:
        src, remote = name.split("__", 1)
        return src, remote
    return path.stem, path.name


def build_record(
    source_name: str,
    source_record_id: str,
    device_name: str,
    component_name: str,
    freq: float,
    anh: float,
    t1: float,
    t2: float,
    temp_mk: float,
    measurement_date: str,
    chip_id: str,
    cooldown_id: str,
    paper_link: str,
    license_text: str,
    raw_source_file: str,
    quality_flags: Sequence[str],
    notes: str,
    source_confidence: float,
    fit_t1_r2: float = np.nan,
    fit_t2_r2: float = np.nan,
    flux_mV: float = np.nan,
) -> Dict[str, Any]:
    return {
        "source_name": source_name,
        "source_record_id": source_record_id,
        "device_name": device_name,
        "component_name": component_name,
        "measured_freq_01_GHz": freq,
        "measured_anharmonicity_GHz": anh,
        "measured_t1_us": t1,
        "measured_t2_us": t2,
        "temperature_mK": temp_mk,
        "measurement_date_utc": measurement_date,
        "chip_id": chip_id,
        "cooldown_id": cooldown_id,
        "paper_link": paper_link,
        "license": license_text,
        "raw_source_file": raw_source_file,
        "quality_flags": ";".join(sorted(set([f for f in quality_flags if f]))),
        "notes": notes,
        "source_confidence": float(np.clip(source_confidence, 0.05, 1.0)),
        "fit_t1_r2": fit_t1_r2,
        "fit_t2_r2": fit_t2_r2,
        "flux_mV": flux_mV,
    }


def parse_squadds(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return rows

    for i, device in enumerate(payload):
        if not isinstance(device, dict):
            continue
        contrib = device.get("contrib_info", {}) if isinstance(device.get("contrib_info"), dict) else {}
        device_name = str(contrib.get("name") or f"device_{i}")
        date_created = str(contrib.get("date_created") or "")
        paper_link = str(device.get("paper_link") or "")

        measured_results = device.get("measured_results", [])
        if not isinstance(measured_results, list):
            continue

        for result in measured_results:
            for path_tuple, comp in iter_dict_nodes(result):
                metrics = extract_component_metrics(comp)
                has_signal = bool(
                    np.isfinite(metrics["measured_t1_us"])
                    or np.isfinite(metrics["measured_t2_us"])
                    or np.isfinite(metrics["measured_freq_01_GHz"])
                )
                if not has_signal:
                    continue

                path_str = "/".join(path_tuple) if path_tuple else "root"
                flags: List[str] = list(metrics["flags"])
                if not np.isfinite(metrics["measured_freq_01_GHz"]):
                    flags.append("missing_freq")
                if not np.isfinite(metrics["measured_t1_us"]) and not np.isfinite(metrics["measured_t2_us"]):
                    flags.append("missing_coherence")
                flags.append("parsed_from_squadds")

                rows.append(
                    build_record(
                        source_name="SQuADDS",
                        source_record_id=f"squadds:{device_name}:{path_str}",
                        device_name=device_name,
                        component_name=path_tuple[-1] if path_tuple else "unknown",
                        freq=metrics["measured_freq_01_GHz"],
                        anh=metrics["measured_anharmonicity_GHz"],
                        t1=metrics["measured_t1_us"],
                        t2=metrics["measured_t2_us"],
                        temp_mk=metrics["temperature_mK"],
                        measurement_date=date_created,
                        chip_id="",
                        cooldown_id="",
                        paper_link=paper_link,
                        license_text="See upstream dataset licensing",
                        raw_source_file=str(path.resolve()),
                        quality_flags=flags,
                        notes=safe_text(device.get("notes") or ""),
                        source_confidence=0.85,
                    )
                )

    return rows


def detect_sep(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    if first.count("\t") >= max(1, first.count(",")):
        return "\t"
    if "," in first:
        return ","
    if ";" in first:
        return ";"
    return r"\s+"


def parse_table_text(text: str) -> pd.DataFrame:
    sep = detect_sep(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", comment="#")
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_fitted_curve_column(col_name: str) -> bool:
    low = str(col_name).lower()
    fitted_markers = (
        "model",
        "fit",
        "fitted",
        "interp",
        "smoothed",
        "simulation",
        "simulated",
    )
    return any(m in low for m in fitted_markers)


def filter_fitted_curve_columns(columns: Sequence[str], include_fitted_curves: bool) -> List[str]:
    if include_fitted_curves:
        return list(columns)
    return [c for c in columns if not is_fitted_curve_column(c)]


def parse_table_dataframe(
    df: pd.DataFrame,
    source_name: str,
    source_record_prefix: str,
    raw_source_file: str,
    notes_suffix: str,
    include_fitted_curves: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    cols = list(df.columns)

    t1_cols = filter_fitted_curve_columns(
        find_cols(cols, ["t1", "relaxation"], excludes=["fit_time", "time"]),
        include_fitted_curves=include_fitted_curves,
    )
    t2_cols = filter_fitted_curve_columns(
        find_cols(cols, ["t2", "dephasing"], excludes=["time"]),
        include_fitted_curves=include_fitted_curves,
    )
    if not t1_cols and not t2_cols:
        return rows

    freq_candidates_all = find_cols(
        cols,
        ["f_01", "f01", "freq_01", "qubit_freq", "qubit frequency", "frequency", "freq", "omega"],
        excludes=["resonator", "cavity", "readout", "drive"],
    )
    freq_candidates = filter_fitted_curve_columns(freq_candidates_all, include_fitted_curves=include_fitted_curves)
    if not freq_candidates and include_fitted_curves:
        freq_candidates = list(freq_candidates_all)
    freq_col = max(freq_candidates, key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum())) if freq_candidates else None

    anh_candidates_all = find_cols(cols, ["anh", "alpha"])
    anh_candidates = filter_fitted_curve_columns(anh_candidates_all, include_fitted_curves=include_fitted_curves)
    if not anh_candidates and include_fitted_curves:
        anh_candidates = list(anh_candidates_all)
    anh_col = max(anh_candidates, key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum())) if anh_candidates else None

    temp_candidates = find_cols(cols, ["temperature", "temp"])
    temp_col = max(temp_candidates, key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum())) if temp_candidates else None

    chip_col = first_col(cols, ["chip_id", "chip", "device_id", "device"])
    cooldown_col = first_col(cols, ["cooldown", "run_id", "experiment_id"])
    date_col = first_col(cols, ["date", "timestamp", "time_utc"])
    flux_col = first_col(cols, ["flux", "bias", "voltage"])
    r2_col = first_col(cols, ["r2", "r_squared", "rsq"])

    pair_map: Dict[str, Dict[str, Any]] = {}
    for c in t1_cols:
        k = component_from_column(c)
        n = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        pair_map.setdefault(k, {"t1": None, "t2": None, "t1_n": -1, "t2_n": -1})
        if n > int(pair_map[k]["t1_n"]):
            pair_map[k]["t1"] = c
            pair_map[k]["t1_n"] = n
    for c in t2_cols:
        k = component_from_column(c)
        n = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        pair_map.setdefault(k, {"t1": None, "t2": None, "t1_n": -1, "t2_n": -1})
        if n > int(pair_map[k]["t2_n"]):
            pair_map[k]["t2"] = c
            pair_map[k]["t2_n"] = n

    for ridx, row in df.iterrows():
        for comp_key, pair in pair_map.items():
            t1 = np.nan
            t2 = np.nan
            fit_t1_r2 = np.nan
            fit_t2_r2 = np.nan

            if pair.get("t1"):
                v, _ = to_float(row.get(pair["t1"]))
                t1 = normalize_time_us(v, str(pair["t1"]))
            if pair.get("t2"):
                v, _ = to_float(row.get(pair["t2"]))
                t2 = normalize_time_us(v, str(pair["t2"]))

            if not (np.isfinite(t1) or np.isfinite(t2)):
                continue

            freq = np.nan
            if freq_col:
                v, _ = to_float(row.get(freq_col))
                freq = normalize_frequency_ghz(v, str(freq_col))

            anh = np.nan
            if anh_col:
                v, _ = to_float(row.get(anh_col))
                anh = normalize_anh_ghz(v, str(anh_col))

            temp = np.nan
            if temp_col:
                v, _ = to_float(row.get(temp_col))
                temp = normalize_temp_mk(v, str(temp_col))

            if r2_col:
                rv, _ = to_float(row.get(r2_col))
                if np.isfinite(rv):
                    if np.isfinite(t1):
                        fit_t1_r2 = float(rv)
                    if np.isfinite(t2):
                        fit_t2_r2 = float(rv)

            flux = np.nan
            if flux_col:
                fv, _ = to_float(row.get(flux_col))
                if np.isfinite(fv):
                    flux = float(fv)

            flags: List[str] = ["parsed_from_table"]
            if not include_fitted_curves:
                flags.append("fitted_curves_excluded")
            if not np.isfinite(freq):
                flags.append("missing_freq")
            if np.isfinite(t1) and t1 < 0.01:
                flags.append("tiny_t1")
            if np.isfinite(t2) and t2 < 0.01:
                flags.append("tiny_t2")

            conf = 0.75
            if np.isfinite(freq):
                conf += 0.10
            if np.isfinite(t1) and np.isfinite(t2):
                conf += 0.05
            if np.isfinite(fit_t1_r2):
                conf += 0.05 * max(0.0, min(1.0, fit_t1_r2))

            comp_name = comp_key if comp_key != "main" else "qubit"
            chip_id = safe_text(row.get(chip_col)) if chip_col else ""
            cooldown_id = safe_text(row.get(cooldown_col)) if cooldown_col else ""
            meas_date = safe_text(row.get(date_col)) if date_col else ""

            notes = notes_suffix
            if pair.get("t1"):
                notes = f"{notes};t1_col={pair.get('t1')}" if notes else f"t1_col={pair.get('t1')}"
            if pair.get("t2"):
                notes = f"{notes};t2_col={pair.get('t2')}" if notes else f"t2_col={pair.get('t2')}"

            rows.append(
                build_record(
                    source_name=source_name,
                    source_record_id=f"{source_record_prefix}:{comp_key}:{ridx}",
                    device_name=chip_id or source_name,
                    component_name=comp_name,
                    freq=freq,
                    anh=anh,
                    t1=t1,
                    t2=t2,
                    temp_mk=temp,
                    measurement_date=meas_date,
                    chip_id=chip_id,
                    cooldown_id=cooldown_id,
                    paper_link="",
                    license_text="See upstream dataset licensing",
                    raw_source_file=raw_source_file,
                    quality_flags=flags,
                    notes=notes,
                    source_confidence=conf,
                    fit_t1_r2=fit_t1_r2,
                    fit_t2_r2=fit_t2_r2,
                    flux_mV=flux,
                )
            )

    return rows


def parse_json_payload(
    payload: Any,
    source_name: str,
    source_record_prefix: str,
    raw_source_file: str,
    notes_suffix: str,
    include_fitted_curves: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path_tuple, node in iter_dict_nodes(payload):
        metrics = extract_component_metrics(node)
        has_signal = bool(np.isfinite(metrics["measured_t1_us"]) or np.isfinite(metrics["measured_t2_us"]))
        if not has_signal:
            continue

        path_str = "/".join(path_tuple) if path_tuple else "root"
        flags = list(metrics["flags"])
        flags.append("parsed_from_json")
        if not np.isfinite(metrics["measured_freq_01_GHz"]):
            flags.append("missing_freq")

        rows.append(
            build_record(
                source_name=source_name,
                source_record_id=f"{source_record_prefix}:{path_str}",
                device_name=source_name,
                component_name=path_tuple[-1] if path_tuple else "unknown",
                freq=metrics["measured_freq_01_GHz"],
                anh=metrics["measured_anharmonicity_GHz"],
                t1=metrics["measured_t1_us"],
                t2=metrics["measured_t2_us"],
                temp_mk=metrics["temperature_mK"],
                measurement_date="",
                chip_id="",
                cooldown_id="",
                paper_link="",
                license_text="See upstream dataset licensing",
                raw_source_file=raw_source_file,
                quality_flags=flags,
                notes=notes_suffix,
                source_confidence=0.65,
            )
        )

    return rows


def parse_excel_bytes(data: bytes) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    bio = io.BytesIO(data)
    try:
        xls = pd.ExcelFile(bio)
    except Exception:
        return out
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        out.append(df)
    return out


def parse_parquet_bytes(data: bytes) -> pd.DataFrame:
    bio = io.BytesIO(data)
    try:
        df = pd.read_parquet(bio)
    except Exception:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_zip(path: Path, source_name: str, source_prefix: str, include_fitted_curves: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_source_file = str(path.resolve())

    with zipfile.ZipFile(path, mode="r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            if member.file_size > 80 * 1024 * 1024:
                continue

            lower = member.filename.lower()
            ext = Path(lower).suffix

            try:
                blob = zf.read(member)
            except Exception:
                continue

            notes = f"zip_member={member.filename}"
            rec_prefix = f"{source_prefix}:{member.filename}"

            if ext in {".csv", ".tsv", ".txt"}:
                text = blob.decode("utf-8", errors="ignore")
                if not text.strip():
                    continue
                df = parse_table_text(text)
                rows.extend(
                    parse_table_dataframe(
                        df=df,
                        source_name=source_name,
                        source_record_prefix=rec_prefix,
                        raw_source_file=raw_source_file,
                        notes_suffix=notes,
                        include_fitted_curves=include_fitted_curves,
                    )
                )
            elif ext == ".json":
                try:
                    payload = json.loads(blob.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                rows.extend(
                    parse_json_payload(
                        payload=payload,
                        source_name=source_name,
                        source_record_prefix=rec_prefix,
                        raw_source_file=raw_source_file,
                        notes_suffix=notes,
                        include_fitted_curves=include_fitted_curves,
                    )
                )
            elif ext in {".xlsx", ".xls"}:
                for sheet_i, df in enumerate(parse_excel_bytes(blob)):
                    rows.extend(
                        parse_table_dataframe(
                            df=df,
                            source_name=source_name,
                            source_record_prefix=f"{rec_prefix}:sheet{sheet_i}",
                            raw_source_file=raw_source_file,
                            notes_suffix=f"{notes};sheet={sheet_i}",
                            include_fitted_curves=include_fitted_curves,
                        )
                    )
            elif ext == ".parquet":
                df = parse_parquet_bytes(blob)
                rows.extend(
                    parse_table_dataframe(
                        df=df,
                        source_name=source_name,
                        source_record_prefix=rec_prefix,
                        raw_source_file=raw_source_file,
                        notes_suffix=notes,
                        include_fitted_curves=include_fitted_curves,
                    )
                )

    return rows


def parse_file(path: Path, include_fitted_curves: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    source_raw, remote_name = source_parts_from_filename(path)
    source_name = human_source_name(source_raw)
    source_prefix = f"{source_raw}:{remote_name}"

    if "measured_device_database.json" in remote_name:
        return parse_squadds(path), "squadds"

    ext = path.suffix.lower()
    raw_source_file = str(path.resolve())

    if ext == ".zip":
        return parse_zip(path=path, source_name=source_name, source_prefix=source_prefix, include_fitted_curves=include_fitted_curves), "zip"

    if ext in {".csv", ".tsv", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        df = parse_table_text(text)
        rows = parse_table_dataframe(
            df=df,
            source_name=source_name,
            source_record_prefix=source_prefix,
            raw_source_file=raw_source_file,
            notes_suffix=f"file={remote_name}",
            include_fitted_curves=include_fitted_curves,
        )
        return rows, "table"

    if ext == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return [], "json"
        rows = parse_json_payload(
            payload=payload,
            source_name=source_name,
            source_record_prefix=source_prefix,
            raw_source_file=raw_source_file,
            notes_suffix=f"file={remote_name}",
            include_fitted_curves=include_fitted_curves,
        )
        return rows, "json"

    if ext in {".xlsx", ".xls"}:
        rows: List[Dict[str, Any]] = []
        blob = path.read_bytes()
        for sheet_i, df in enumerate(parse_excel_bytes(blob)):
            rows.extend(
                parse_table_dataframe(
                    df=df,
                    source_name=source_name,
                    source_record_prefix=f"{source_prefix}:sheet{sheet_i}",
                    raw_source_file=raw_source_file,
                    notes_suffix=f"file={remote_name};sheet={sheet_i}",
                    include_fitted_curves=include_fitted_curves,
                )
            )
        return rows, "excel"

    if ext == ".parquet":
        df = parse_parquet_bytes(path.read_bytes())
        rows = parse_table_dataframe(
            df=df,
            source_name=source_name,
            source_record_prefix=source_prefix,
            raw_source_file=raw_source_file,
            notes_suffix=f"file={remote_name}",
            include_fitted_curves=include_fitted_curves,
        )
        return rows, "parquet"

    return [], "ignored"


def finalize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    df = pd.DataFrame(rows)
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

    num_cols = [
        "measured_freq_01_GHz",
        "measured_anharmonicity_GHz",
        "measured_t1_us",
        "measured_t2_us",
        "temperature_mK",
        "source_confidence",
        "fit_t1_r2",
        "fit_t2_r2",
        "flux_mV",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    has_any_label = df["measured_t1_us"].notna() | df["measured_t2_us"].notna()
    df = df[has_any_label].copy()

    if "source_record_id" in df.columns:
        df = df.drop_duplicates(subset=["source_record_id"], keep="first")

    df = df.sort_values(["source_name", "source_record_id"]).reset_index(drop=True)
    return df.loc[:, list(CANONICAL_COLUMNS)]


def main() -> int:
    args = parse_args()

    if not args.bronze_dir.exists():
        raise SystemExit(f"Bronze dir not found: {args.bronze_dir}")

    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    parser_counts: Dict[str, int] = {}
    file_rows: List[Dict[str, Any]] = []

    all_files = sorted([p for p in args.bronze_dir.glob("*") if p.is_file()])
    prefixed_remote_names = set()
    for p in all_files:
        if "__" in p.name:
            _, remote = source_parts_from_filename(p)
            prefixed_remote_names.add(remote)

    files: List[Path] = []
    for p in all_files:
        if "__" not in p.name and p.name in prefixed_remote_names:
            continue
        files.append(p)

    if args.strict and not files:
        raise SystemExit("No bronze files found")

    for p in files:
        parsed_rows, parser_name = parse_file(p, include_fitted_curves=args.include_fitted_curves)
        parser_counts[parser_name] = int(parser_counts.get(parser_name, 0) + 1)

        src_raw, _ = source_parts_from_filename(p)
        src_name = human_source_name(src_raw)
        source_counts[src_name] = int(source_counts.get(src_name, 0) + len(parsed_rows))

        file_rows.append(
            {
                "file": str(p.resolve()),
                "parser": parser_name,
                "rows_parsed": int(len(parsed_rows)),
            }
        )
        rows.extend(parsed_rows)

    out_df = finalize(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bronze_dir": str(args.bronze_dir.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "rows_total": int(len(out_df)),
        "rows_with_t1": int(out_df["measured_t1_us"].notna().sum()) if len(out_df) > 0 else 0,
        "rows_with_t2": int(out_df["measured_t2_us"].notna().sum()) if len(out_df) > 0 else 0,
        "rows_with_freq": int(out_df["measured_freq_01_GHz"].notna().sum()) if len(out_df) > 0 else 0,
        "sources": source_counts,
        "parser_file_counts": parser_counts,
        "include_fitted_curves": bool(args.include_fitted_curves),
        "file_rows": file_rows,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Public Canonical Dataset Build Complete ===")
    print(
        f"rows_total={report['rows_total']} rows_with_t1={report['rows_with_t1']} "
        f"rows_with_t2={report['rows_with_t2']} rows_with_freq={report['rows_with_freq']}"
    )
    print(f"output={args.output_csv}")
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
