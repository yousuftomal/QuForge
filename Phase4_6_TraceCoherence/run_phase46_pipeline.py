#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run complete Phase 4.6 public-trace bootstrap pipeline")
    parser.add_argument("--python", type=Path, default=root / "Dataset" / ".venv310" / "Scripts" / "python.exe")
    parser.add_argument("--max-distance", type=float, default=1.6)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--measured-weight", type=float, default=1.0)
    parser.add_argument("--proxy-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-label-blend", type=float, default=0.35)
    parser.add_argument("--synthetic-regularization-weight", type=float, default=0.75)
    parser.add_argument("--neighbors-per-anchor", type=int, default=24)
    return parser.parse_args()


def run_step(py: Path, root: Path, args: list[str]) -> None:
    cmd = [str(py), *args]
    print("\n>>>", " ".join(args))
    subprocess.run(cmd, cwd=root, check=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = args.python

    run_step(py, root, [
        "Phase4_6_TraceCoherence/augment_with_zenodo_8004359_traces.py",
    ])

    run_step(py, root, [
        "Phase4_5_PublicData/map_public_to_internal.py",
        "--public-csv", "Dataset/public_sources/silver/public_measurements_canonical_augmented.csv",
        "--max-distance", str(args.max_distance),
        "--min-confidence", str(args.min_confidence),
    ])

    run_step(py, root, [
        "Phase4_5_PublicData/augment_anchor_conditioned_regularization.py",
        "--measurement-csv", "Dataset/measurement_dataset_public_bootstrap.csv",
        "--output-csv", "Dataset/measurement_dataset_public_bootstrap_augmented.csv",
        "--report-json", "Dataset/measurement_dataset_public_bootstrap_augmented.report.json",
        "--neighbors-per-anchor", str(args.neighbors_per_anchor),
    ])

    run_step(py, root, [
        "Phase4_Coherence/train_phase4_coherence.py",
        "--label-mode", "hybrid",
        "--measurement-csv", "Dataset/measurement_dataset_public_bootstrap_augmented.csv",
        "--measured-weight", str(args.measured_weight),
        "--proxy-weight", str(args.proxy_weight),
        "--synthetic-label-blend", str(args.synthetic_label_blend),
        "--synthetic-regularization-weight", str(args.synthetic_regularization_weight),
    ])

    run_step(py, root, [
        "Phase4_Coherence/validate_phase4_coherence.py",
        "--measurement-csv", "Dataset/measurement_dataset_public_bootstrap_augmented.csv",
    ])

    print("\n=== Phase 4.6 Pipeline Complete ===")
    print("Outputs:")
    print("- Dataset/public_sources/silver/public_measurements_canonical_augmented.csv")
    print("- Dataset/measurement_dataset_public_bootstrap.csv")
    print("- Dataset/measurement_dataset_public_bootstrap_augmented.csv")
    print("- Phase4_Coherence/artifacts/phase4_training_summary.json")
    print("- Phase4_Coherence/artifacts/phase4_validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
