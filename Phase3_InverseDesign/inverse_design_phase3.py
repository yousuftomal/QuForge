#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine import build_target_map, infer_active_target_cols, load_context, run_inverse_design


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 3 inverse design engine")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--phase2-bundle", type=Path, default=root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument("--phase1-model", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-meta", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")

    parser.add_argument("--freq-01-ghz", type=float, required=True)
    parser.add_argument("--anharmonicity-ghz", type=float, required=True)
    parser.add_argument("--ej-ghz", type=float, default=None)
    parser.add_argument("--ec-ghz", type=float, default=None)
    parser.add_argument("--charge-sensitivity-ghz", type=float, default=None)

    parser.add_argument("--library-split", choices=("train", "all"), default="train")
    parser.add_argument("--retrieval-top-k", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--population-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--elite-fraction", type=float, default=0.22)
    parser.add_argument("--explore-fraction", type=float, default=0.12)
    parser.add_argument("--start-mutation-scale", type=float, default=0.15)
    parser.add_argument("--end-mutation-scale", type=float, default=0.02)

    parser.add_argument("--surrogate-weight", type=float, default=1.0)
    parser.add_argument("--embedding-weight", type=float, default=0.0)
    parser.add_argument("--robustness-weight", type=float, default=0.0)
    parser.add_argument("--robust-samples", type=int, default=0)
    parser.add_argument("--fabrication-tolerance", type=float, default=0.05)
    parser.add_argument("--robust-std-weight", type=float, default=0.30)

    parser.add_argument("--bounds-low-q", type=float, default=0.001)
    parser.add_argument("--bounds-high-q", type=float, default=0.999)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ctx = load_context(
        root=root,
        single_csv=args.single_csv,
        phase2_bundle=args.phase2_bundle,
        phase1_model_path=args.phase1_model,
        phase1_meta_path=args.phase1_meta,
    )

    target = build_target_map(
        ctx,
        freq_01_GHz=args.freq_01_ghz,
        anharmonicity_GHz=args.anharmonicity_ghz,
        EJ_GHz=args.ej_ghz,
        EC_GHz=args.ec_ghz,
        charge_sensitivity_GHz=args.charge_sensitivity_ghz,
    )
    active_cols = infer_active_target_cols(
        ej_given=args.ej_ghz is not None,
        ec_given=args.ec_ghz is not None,
        charge_given=args.charge_sensitivity_ghz is not None,
    )

    payload = run_inverse_design(
        ctx,
        target_map=target,
        active_target_cols=active_cols,
        library_split=args.library_split,
        retrieval_top_k=args.retrieval_top_k,
        top_n=args.top_n,
        population_size=args.population_size,
        iterations=args.iterations,
        elite_fraction=args.elite_fraction,
        explore_fraction=args.explore_fraction,
        start_mutation_scale=args.start_mutation_scale,
        end_mutation_scale=args.end_mutation_scale,
        surrogate_weight=args.surrogate_weight,
        embedding_weight=args.embedding_weight,
        robustness_weight=args.robustness_weight,
        robust_samples=args.robust_samples,
        fabrication_tolerance=args.fabrication_tolerance,
        robust_std_weight=args.robust_std_weight,
        bounds_low_q=args.bounds_low_q,
        bounds_high_q=args.bounds_high_q,
        random_state=args.random_state,
    )

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

