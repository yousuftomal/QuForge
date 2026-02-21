#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import urllib.parse
import urllib.request
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DIRECT_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "squadds_measured_database",
        "group": "squadds",
        "url": "https://huggingface.co/datasets/SQuADDS/SQuADDS_DB/resolve/main/measured_device_database.json",
        "filename": "squadds_measured_device_database.json",
        "is_large": False,
    },
    {
        "name": "squadds_qubit_half_wave_cavity",
        "group": "squadds",
        "url": "https://huggingface.co/datasets/SQuADDS/SQuADDS_DB/resolve/main/qubit_half-wave-cavity_df.parquet",
        "filename": "squadds_qubit_half-wave-cavity_df.parquet",
        "is_large": False,
    },
]

ZENODO_RECORD_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "zenodo_4336924_twin_qubit",
        "group": "zenodo",
        "record_id": 4336924,
        "is_large": False,
    },
    {
        "name": "zenodo_8004359_ista_transmon",
        "group": "zenodo",
        "record_id": 8004359,
        "is_large": True,
    },
    {
        "name": "zenodo_18045662_daqec_benchmark",
        "group": "zenodo",
        "record_id": 18045662,
        "is_large": False,
    },
    {
        "name": "zenodo_15364358_low_latency_qec",
        "group": "zenodo",
        "record_id": 15364358,
        "is_large": True,
    },
    {
        "name": "zenodo_13808824_leakage_control",
        "group": "zenodo",
        "record_id": 13808824,
        "is_large": False,
    },
    {
        "name": "zenodo_14628434_optical_control",
        "group": "zenodo",
        "record_id": 14628434,
        "is_large": False,
    },
]

CKAN_PACKAGE_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "data_gov_mds2_2516_jpg_3k_control",
        "group": "data_gov",
        "package_id": "digital-control-of-a-superconducting-qubit-using-a-josephson-pulse-generator-at-3-k",
        "is_large": False,
    },
    {
        "name": "data_gov_mds2_2932_coherence_limited_control",
        "group": "data_gov",
        "package_id": "coherence-limited-digital-control-of-a-superconducting-qubit-using-a-josephson-pulse-gener",
        "is_large": False,
    },
    {
        "name": "data_gov_mds2_3027_parametric_dispersive",
        "group": "data_gov",
        "package_id": "data-for-nature-physics-manuscript-strong-parametric-dispersive-shifts-in-a-statically-dec",
        "is_large": False,
    },
    {
        "name": "data_gov_mds2_2756_entangling_over_fiber",
        "group": "data_gov",
        "package_id": "entangling-superconducting-qubits-over-optical-fiber-towards-optimization-and-implementati",
        "is_large": False,
    },
]

ALLOWED_EXTENSIONS = {
    ".zip",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".parquet",
    ".xlsx",
    ".xls",
    ".h5",
    ".hdf5",
    ".md",
    ".yaml",
    ".yml",
}

FORMAT_EXTENSION_HINTS = {
    "csv": ".csv",
    "tsv": ".tsv",
    "txt": ".txt",
    "text": ".txt",
    "json": ".json",
    "zip": ".zip",
    "xls": ".xls",
    "xlsx": ".xlsx",
    "parquet": ".parquet",
    "h5": ".h5",
    "hdf5": ".hdf5",
}


class FetchError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fetch public superconducting-qubit datasets into Dataset/public_sources/bronze"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "Dataset" / "public_sources" / "bronze")
    parser.add_argument("--report-json", type=Path, default=root / "Dataset" / "public_sources" / "fetch_report.json")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-download-mb", type=float, default=120.0)
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Allow groups tagged large (e.g., Zenodo records with very large assets)",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even if they already exist")

    parser.add_argument("--skip-squadds", action="store_true")
    parser.add_argument("--skip-zenodo-small", action="store_true")
    parser.add_argument("--skip-zenodo-large", action="store_true")

    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--skip-data-gov", action="store_true")
    parser.add_argument("--source-filter", type=str, default="", help="Regex filter on source names")
    return parser.parse_args()


def sanitize_filename(name: str, default: str = "file") -> str:
    s = name.strip().replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._")
    return s if s else default


def to_json(url: str, timeout_sec: int) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FetchError(f"invalid json from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FetchError(f"json root is not object from {url}")
    return payload


def url_basename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = Path(parsed.path).name
    return base or "download"


def canonicalize_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe='/-._~')
    safe_query = parts.query
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, safe_path, safe_query, parts.fragment))


def infer_extension(url: str, name_hint: str, format_hint: str) -> str:
    for candidate in [name_hint, url_basename(url)]:
        ext = Path(candidate).suffix.lower()
        if ext:
            return ext

    fmt = str(format_hint).strip().lower()
    for key, ext in FORMAT_EXTENSION_HINTS.items():
        if key in fmt:
            return ext
    return ""


def make_local_filename(source_name: str, remote_name: str, fallback_ext: str = "") -> str:
    remote_base = sanitize_filename(Path(remote_name).name or remote_name or "file")
    ext = Path(remote_base).suffix
    if not ext and fallback_ext:
        remote_base = f"{remote_base}{fallback_ext}"
    return f"{sanitize_filename(source_name)}__{remote_base}"


def keep_extension(ext: str) -> bool:
    if not ext:
        return False
    return ext.lower() in ALLOWED_EXTENSIONS


def expand_direct_sources(timeout_sec: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for src in DIRECT_SOURCES:
        items.append(
            {
                "source_name": src["name"],
                "source_group": src["group"],
                "source_kind": "direct",
                "is_large": bool(src.get("is_large", False)),
                "remote_name": src["filename"],
                "url": canonicalize_url(src["url"]),
                "filename": make_local_filename(src["name"], src["filename"]),
                "record_ref": src["url"],
            }
        )
    return items


def expand_zenodo_sources(timeout_sec: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for src in ZENODO_RECORD_SOURCES:
        rid = int(src["record_id"])
        record_url = f"https://zenodo.org/api/records/{rid}"
        payload = to_json(record_url, timeout_sec=timeout_sec)
        files = payload.get("files", [])
        if not isinstance(files, list):
            files = []
        for f in files:
            if not isinstance(f, dict):
                continue
            key = str(f.get("key") or "")
            links = f.get("links") if isinstance(f.get("links"), dict) else {}
            content_url = str(links.get("content") or links.get("self") or "")
            if not key or not content_url:
                continue
            ext = Path(key).suffix.lower()
            if not keep_extension(ext):
                continue

            items.append(
                {
                    "source_name": src["name"],
                    "source_group": src["group"],
                    "source_kind": "zenodo_record",
                    "is_large": bool(src.get("is_large", False)),
                    "remote_name": key,
                    "url": canonicalize_url(content_url),
                    "filename": make_local_filename(src["name"], key),
                    "record_ref": f"zenodo:{rid}",
                    "record_id": rid,
                }
            )
    return items


def expand_ckan_sources(timeout_sec: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    base = "https://catalog.data.gov/api/3/action/package_show?id="

    for src in CKAN_PACKAGE_SOURCES:
        package_id = str(src["package_id"])
        payload = to_json(base + urllib.parse.quote(package_id, safe=""), timeout_sec=timeout_sec)
        if not bool(payload.get("success")):
            raise FetchError(f"ckan package_show failed for {package_id}: {payload.get('error')}")

        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        resources = result.get("resources") if isinstance(result.get("resources"), list) else []

        for res in resources:
            if not isinstance(res, dict):
                continue
            url = str(res.get("url") or "")
            if not url:
                continue
            name_hint = str(res.get("name") or "resource")
            format_hint = str(res.get("format") or "")
            ext = infer_extension(url=url, name_hint=name_hint, format_hint=format_hint)
            if not keep_extension(ext):
                continue

            remote_name = Path(url_basename(url)).name
            if not remote_name or "." not in remote_name:
                remote_name = sanitize_filename(name_hint, default="resource") + ext

            items.append(
                {
                    "source_name": src["name"],
                    "source_group": src["group"],
                    "source_kind": "ckan_package",
                    "is_large": bool(src.get("is_large", False)),
                    "remote_name": remote_name,
                    "url": canonicalize_url(url),
                    "filename": make_local_filename(src["name"], remote_name, fallback_ext=ext),
                    "record_ref": f"ckan:{package_id}",
                    "package_id": package_id,
                    "resource_format": format_hint,
                }
            )
    return items


def get_content_length(url: str, timeout_sec: int) -> int:
    safe_url = canonicalize_url(url)
    req = urllib.request.Request(url=safe_url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            header = resp.headers.get("Content-Length")
    except Exception:
        return -1

    if header is None:
        return -1
    try:
        return int(header)
    except ValueError:
        return -1


def download_file(url: str, output_path: Path, timeout_sec: int, max_retries: int = 3) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_url = canonicalize_url(url)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url=safe_url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                with tmp_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            os.replace(tmp_path, output_path)
            return int(output_path.stat().st_size)
        except Exception as exc:
            last_exc = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            if attempt + 1 < max_retries:
                time.sleep(1.0 + attempt)
                continue
            raise

    raise RuntimeError(f"download failed for {safe_url}: {last_exc}")


def source_allowed(item: Dict[str, Any], args: argparse.Namespace) -> bool:
    source_name = str(item.get("source_name", ""))
    source_group = str(item.get("source_group", ""))

    if args.source_filter:
        if re.search(args.source_filter, source_name) is None:
            return False

    if source_group == "squadds" and args.skip_squadds:
        return False

    if source_group == "zenodo":
        if args.skip_zenodo:
            return False
        if source_name == "zenodo_4336924_twin_qubit" and args.skip_zenodo_small:
            return False
        if bool(item.get("is_large", False)) and args.skip_zenodo_large:
            return False

    if source_group == "data_gov" and args.skip_data_gov:
        return False

    return True


def unique_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        key = f"{it.get('source_name','')}::{it.get('filename','')}::{it.get('url','')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_download_mb * 1024 * 1024)

    rows: List[Dict[str, object]] = []
    expansion_errors: List[Dict[str, str]] = []

    candidates: List[Dict[str, Any]] = []
    candidates.extend(expand_direct_sources(timeout_sec=args.timeout_sec))

    try:
        candidates.extend(expand_zenodo_sources(timeout_sec=args.timeout_sec))
    except Exception as exc:
        expansion_errors.append({"source_group": "zenodo", "error": str(exc)})

    try:
        candidates.extend(expand_ckan_sources(timeout_sec=args.timeout_sec))
    except Exception as exc:
        expansion_errors.append({"source_group": "data_gov", "error": str(exc)})

    candidates = unique_items(candidates)

    for item in candidates:
        source_name = str(item["source_name"])
        url = str(item["url"])
        filename = str(item["filename"])
        is_large = bool(item.get("is_large", False))

        if not source_allowed(item, args):
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "skipped_by_flag",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                }
            )
            continue

        if is_large and not args.include_large:
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "skipped_large_source",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                    "reason": "use --include-large to enable",
                }
            )
            continue

        out_path = args.output_dir / filename
        if out_path.exists() and not args.force:
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "cached",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                    "bytes_downloaded": int(out_path.stat().st_size),
                    "path": str(out_path.resolve()),
                }
            )
            continue

        content_length = get_content_length(url=url, timeout_sec=args.timeout_sec)
        if content_length > 0 and content_length > max_bytes:
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "skipped_size_limit",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                    "content_length_bytes": content_length,
                    "max_allowed_bytes": max_bytes,
                }
            )
            continue

        try:
            bytes_written = download_file(url=url, output_path=out_path, timeout_sec=args.timeout_sec)
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "downloaded",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                    "bytes_downloaded": bytes_written,
                    "content_length_bytes": content_length,
                    "path": str(out_path.resolve()),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source_name": source_name,
                    "source_group": item.get("source_group"),
                    "source_kind": item.get("source_kind"),
                    "status": "error",
                    "url": canonicalize_url(url),
                    "filename": filename,
                    "is_large": is_large,
                    "record_ref": item.get("record_ref"),
                    "content_length_bytes": content_length,
                    "error": str(exc),
                }
            )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir.resolve()),
        "max_download_mb": float(args.max_download_mb),
        "include_large": bool(args.include_large),
        "expansion_errors": expansion_errors,
        "rows": rows,
        "summary": {
            "candidates": int(len(candidates)),
            "downloaded": int(sum(1 for r in rows if r["status"] == "downloaded")),
            "cached": int(sum(1 for r in rows if r["status"] == "cached")),
            "skipped": int(sum(1 for r in rows if str(r["status"]).startswith("skipped"))),
            "errors": int(sum(1 for r in rows if r["status"] == "error")),
            "by_source": {
                str(k): int(v)
                for k, v in (
                    Counter(str(r.get("source_name", "")) for r in rows).items()
                )
            },
        },
    }

    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Public Dataset Fetch Complete ===")
    print(
        f"candidates={report['summary']['candidates']} "
        f"downloaded={report['summary']['downloaded']} "
        f"cached={report['summary']['cached']} "
        f"skipped={report['summary']['skipped']} "
        f"errors={report['summary']['errors']}"
    )
    print(f"report={args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
