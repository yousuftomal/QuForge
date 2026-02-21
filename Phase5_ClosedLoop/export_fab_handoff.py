#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "Phase5_ClosedLoop" / "artifacts"
    parser = argparse.ArgumentParser(description="Export fabrication handoff package from Phase 5 selected candidates")
    parser.add_argument("--selected-csv", type=Path, default=artifacts / "phase5_selected_candidates.csv")
    parser.add_argument("--summary-json", type=Path, default=artifacts / "phase5_summary.json")
    parser.add_argument("--output-dir", type=Path, default=artifacts / "handoff")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--require-pass-all", action="store_true", default=True)
    parser.add_argument("--allow-ood", action="store_true")
    parser.add_argument("--allow-low-confidence", action="store_true")
    parser.add_argument("--fallback-if-empty", action="store_true", default=True)
    return parser.parse_args()


def bool_val(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def safe_float(v: Any) -> float:
    try:
        if pd.isna(v):
            return float("nan")
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float("nan")


def build_risk_flags(row: pd.Series) -> List[str]:
    flags: List[str] = []
    if bool_val(row.get("coh_combined_ood")):
        flags.append("ood")

    conf = str(row.get("coh_confidence", "")).strip().lower()
    if conf == "low":
        flags.append("low_confidence")
    elif conf == "medium":
        flags.append("medium_confidence")

    if not bool_val(row.get("pass_all", False)):
        flags.append("failed_gates")

    robust = safe_float(row.get("robust_mean_error"))
    if np.isfinite(robust) and robust > 0.12:
        flags.append("high_robust_error")

    return flags


def candidate_record(row: pd.Series, handoff_rank: int) -> Dict[str, Any]:
    rec = {
        "handoff_rank": int(handoff_rank),
        "target_id": str(row.get("target_id", "")),
        "source_rank": int(safe_float(row.get("rank"))) if np.isfinite(safe_float(row.get("rank"))) else None,
        "geometry": {
            "pad_width_um": safe_float(row.get("pad_width_um")),
            "pad_height_um": safe_float(row.get("pad_height_um")),
            "gap_um": safe_float(row.get("gap_um")),
            "junction_area_um2": safe_float(row.get("junction_area_um2")),
        },
        "predicted_physics": {
            "freq_01_GHz": safe_float(row.get("pred_freq_01_GHz")),
            "anharmonicity_GHz": safe_float(row.get("pred_anharmonicity_GHz")),
            "EJ_GHz": safe_float(row.get("pred_EJ_GHz")),
            "EC_GHz": safe_float(row.get("pred_EC_GHz")),
            "charge_sensitivity_GHz": safe_float(row.get("pred_charge_sensitivity_GHz")),
        },
        "predicted_coherence": {
            "t1_p10_us": safe_float(row.get("coh_t1_p10_us")),
            "t1_p50_us": safe_float(row.get("coh_t1_p50_us")),
            "t1_p90_us": safe_float(row.get("coh_t1_p90_us")),
            "t2_p10_us": safe_float(row.get("coh_t2_p10_us")),
            "t2_p50_us": safe_float(row.get("coh_t2_p50_us")),
            "t2_p90_us": safe_float(row.get("coh_t2_p90_us")),
        },
        "quality": {
            "phase5_score": safe_float(row.get("phase5_score")),
            "objective_total": safe_float(row.get("objective_total")),
            "surrogate_error": safe_float(row.get("surrogate_error")),
            "robust_mean_error": safe_float(row.get("robust_mean_error")),
            "robust_std_error": safe_float(row.get("robust_std_error")),
            "coh_confidence": str(row.get("coh_confidence", "")),
            "coh_combined_ood": bool_val(row.get("coh_combined_ood")),
            "pass_all": bool_val(row.get("pass_all", False)),
            "risk_flags": build_risk_flags(row),
        },
    }
    return rec


def main() -> int:
    args = parse_args()

    if not args.selected_csv.exists():
        raise SystemExit(f"selected-csv not found: {args.selected_csv}")

    df = pd.read_csv(args.selected_csv)
    if df.empty:
        raise SystemExit("selected-csv is empty")

    required = [
        "pad_width_um",
        "pad_height_um",
        "gap_um",
        "junction_area_um2",
        "coh_t1_p10_us",
        "coh_t1_p50_us",
        "coh_t1_p90_us",
        "phase5_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"selected-csv missing required columns: {missing}")

    gated = df.copy()

    if args.require_pass_all and "pass_all" in gated.columns:
        gated = gated[gated["pass_all"].apply(bool_val)]

    if not args.allow_ood and "coh_combined_ood" in gated.columns:
        gated = gated[~gated["coh_combined_ood"].apply(bool_val)]

    if not args.allow_low_confidence and "coh_confidence" in gated.columns:
        gated = gated[gated["coh_confidence"].astype(str).str.lower() != "low"]

    gated = gated.sort_values("phase5_score", ascending=False).reset_index(drop=True)

    used_fallback = False
    if len(gated) == 0 and args.fallback_if_empty:
        used_fallback = True
        gated = df.sort_values("phase5_score", ascending=False).reset_index(drop=True)

    top_n = max(1, int(args.top_n))
    exported = gated.head(top_n).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = [
        "target_id",
        "rank",
        "pad_width_um",
        "pad_height_um",
        "gap_um",
        "junction_area_um2",
        "pred_freq_01_GHz",
        "pred_anharmonicity_GHz",
        "pred_EJ_GHz",
        "pred_EC_GHz",
        "pred_charge_sensitivity_GHz",
        "coh_t1_p10_us",
        "coh_t1_p50_us",
        "coh_t1_p90_us",
        "coh_t2_p10_us",
        "coh_t2_p50_us",
        "coh_t2_p90_us",
        "coh_confidence",
        "coh_combined_ood",
        "pass_all",
        "phase5_score",
        "objective_total",
        "surrogate_error",
        "robust_mean_error",
        "robust_std_error",
    ]
    csv_keep = [c for c in csv_cols if c in exported.columns]
    csv_out = exported.loc[:, csv_keep].copy()
    csv_out.insert(0, "handoff_rank", np.arange(1, len(csv_out) + 1))
    csv_path = args.output_dir / "fab_handoff_candidates.csv"
    csv_out.to_csv(csv_path, index=False)

    candidates_json = [candidate_record(r, i + 1) for i, (_, r) in enumerate(exported.iterrows())]

    summary_payload: Dict[str, Any] = {}
    if args.summary_json.exists():
        try:
            summary_payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = {}

    package = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_selected_csv": str(args.selected_csv.resolve()),
        "source_summary_json": str(args.summary_json.resolve()) if args.summary_json.exists() else None,
        "rules": {
            "require_pass_all": bool(args.require_pass_all),
            "allow_ood": bool(args.allow_ood),
            "allow_low_confidence": bool(args.allow_low_confidence),
            "top_n": int(top_n),
        },
        "counts": {
            "selected_input": int(len(df)),
            "eligible_after_gates": int(len(gated)),
            "exported": int(len(exported)),
            "used_fallback": bool(used_fallback),
        },
        "upstream_summary": summary_payload,
        "candidates": candidates_json,
    }

    json_path = args.output_dir / "fab_handoff_package.json"
    json_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# Fabrication Handoff")
    md_lines.append("")
    md_lines.append(f"Generated: {package['generated_at_utc']}")
    md_lines.append("")
    md_lines.append("## Counts")
    md_lines.append("")
    md_lines.append(f"- Input selected rows: {package['counts']['selected_input']}")
    md_lines.append(f"- Eligible after gates: {package['counts']['eligible_after_gates']}")
    md_lines.append(f"- Exported rows: {package['counts']['exported']}")
    md_lines.append(f"- Fallback used: {package['counts']['used_fallback']}")
    md_lines.append("")
    md_lines.append("## Export Files")
    md_lines.append("")
    md_lines.append(f"- `{csv_path}`")
    md_lines.append(f"- `{json_path}`")
    md_lines.append("")
    md_lines.append("## Top Handoff Candidates")
    md_lines.append("")

    for rec in candidates_json[:10]:
        q = rec["quality"]
        c = rec["predicted_coherence"]
        md_lines.append(
            "- "
            f"rank={rec['handoff_rank']} target={rec['target_id']} "
            f"t1_p10={c['t1_p10_us']:.6f}us score={q['phase5_score']:.6f} "
            f"confidence={q['coh_confidence']} risks={','.join(q['risk_flags']) if q['risk_flags'] else 'none'}"
        )

    md_path = args.output_dir / "fab_handoff_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("=== Phase 5.1 Fabrication Handoff Export Complete ===")
    print(f"input_selected={len(df)} eligible={len(gated)} exported={len(exported)} fallback={used_fallback}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())