#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np


def default_artifact_paths(mode: str) -> tuple[Path, Path]:
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    return (artifact_dir / f"{mode}_surrogate.joblib", artifact_dir / f"{mode}_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained Phase-1 surrogate models")
    parser.add_argument("--mode", choices=("single", "coupled"), required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)

    # Single inputs
    parser.add_argument("--pad-width-um", type=float, default=None)
    parser.add_argument("--pad-height-um", type=float, default=None)
    parser.add_argument("--gap-um", type=float, default=None)
    parser.add_argument("--junction-area-um2", type=float, default=None)

    # Coupled inputs
    parser.add_argument("--freq-q1-ghz", type=float, default=None)
    parser.add_argument("--freq-q2-ghz", type=float, default=None)
    parser.add_argument("--resonator-freq-ghz", type=float, default=None)
    parser.add_argument("--coupling-strength-g-ghz", type=float, default=None)
    return parser.parse_args()


def build_input_row(args: argparse.Namespace, feature_cols: List[str]) -> np.ndarray:
    if args.mode == "single":
        raw: Dict[str, float | None] = {
            "pad_width_um": args.pad_width_um,
            "pad_height_um": args.pad_height_um,
            "gap_um": args.gap_um,
            "junction_area_um2": args.junction_area_um2,
        }
    else:
        raw = {
            "freq_q1_GHz": args.freq_q1_ghz,
            "freq_q2_GHz": args.freq_q2_ghz,
            "resonator_freq_GHz": args.resonator_freq_ghz,
            "coupling_strength_g_GHz": args.coupling_strength_g_ghz,
        }

    missing = [k for k, v in raw.items() if v is None]
    if missing:
        raise SystemExit(f"Missing required inputs for mode={args.mode}: {missing}")

    try:
        values = [float(raw[c]) for c in feature_cols]
    except KeyError as exc:
        raise SystemExit(f"Feature mismatch between metadata and CLI inputs: {exc}") from exc

    return np.array(values, dtype=float).reshape(1, -1)


def main() -> int:
    args = parse_args()
    default_model, default_meta = default_artifact_paths(args.mode)
    model_path = args.model_path or default_model
    metadata_path = args.metadata_path or default_meta

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Metadata not found: {metadata_path}")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_cols = metadata["feature_cols"]
    target_cols = metadata["target_cols"]

    x = build_input_row(args=args, feature_cols=feature_cols)
    y_pred = model.predict(x).reshape(-1)

    result = {target_cols[i]: float(y_pred[i]) for i in range(len(target_cols))}
    print(json.dumps({"mode": args.mode, "inputs": dict(zip(feature_cols, x.flatten().tolist())), "prediction": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

