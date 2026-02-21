#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from engine import baseline_rerank_choice, build_target_map, load_context, run_inverse_design


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Phase 3 inverse design engine")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--phase2-bundle", type=Path, default=root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument("--phase1-model", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-meta", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    parser.add_argument("--max-queries-per-split", type=int, default=50)
    parser.add_argument("--library-split", choices=("train", "all"), default="train")
    parser.add_argument("--retrieval-top-k", type=int, default=80)

    parser.add_argument("--population-size", type=int, default=72)
    parser.add_argument("--iterations", type=int, default=30)
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

    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def eval_split(
    split_name: str,
    positions: np.ndarray,
    *,
    ctx,
    args,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if len(positions) == 0:
        return {"split": split_name, "queries": 0}

    qn = min(args.max_queries_per_split, len(positions))
    chosen = rng.choice(positions, size=qn, replace=False)

    phase3_freq_err: List[float] = []
    phase3_anh_err: List[float] = []
    phase3_norm_err: List[float] = []

    baseline_freq_err: List[float] = []
    baseline_anh_err: List[float] = []
    baseline_norm_err: List[float] = []

    elapsed: List[float] = []

    active_cols = tuple(ctx.shared_target_cols)
    freq_idx = active_cols.index("freq_01_GHz")
    anh_idx = active_cols.index("anharmonicity_GHz")
    active_scales = np.asarray(
        [ctx.shared_scales[ctx.shared_target_cols.index(c)] for c in active_cols],
        dtype=float,
    )

    for i, pos in enumerate(chosen, start=1):
        target_row = ctx.xp_raw[int(pos)]
        target = build_target_map(
            ctx,
            freq_01_GHz=float(target_row[ctx.physics_cols.index("freq_01_GHz")]),
            anharmonicity_GHz=float(target_row[ctx.physics_cols.index("anharmonicity_GHz")]),
            EJ_GHz=float(target_row[ctx.physics_cols.index("EJ_GHz")]),
            EC_GHz=float(target_row[ctx.physics_cols.index("EC_GHz")]),
            charge_sensitivity_GHz=float(target_row[ctx.physics_cols.index("charge_sensitivity_GHz")]),
        )

        t0 = time.perf_counter()
        out = run_inverse_design(
            ctx,
            target_map=target,
            active_target_cols=active_cols,
            library_split=args.library_split,
            retrieval_top_k=args.retrieval_top_k,
            top_n=1,
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
            random_state=args.random_state + int(pos),
        )
        elapsed.append(time.perf_counter() - t0)

        best = out["results"][0]
        pred_vec = np.asarray([best[f"pred_{c}"] for c in active_cols], dtype=float)
        tgt_vec = np.asarray([target[c] for c in active_cols], dtype=float)
        diff = (pred_vec - tgt_vec) / active_scales

        phase3_freq_err.append(abs(float(pred_vec[freq_idx] - tgt_vec[freq_idx])))
        phase3_anh_err.append(abs(float(pred_vec[anh_idx] - tgt_vec[anh_idx])))
        phase3_norm_err.append(float(np.sqrt(np.mean(np.square(diff)))))

        base = baseline_rerank_choice(
            ctx,
            target_map=target,
            active_target_cols=active_cols,
            library_split=args.library_split,
            retrieval_top_k=args.retrieval_top_k,
        )
        b_pred_vec = np.asarray([base[f"physics_{c}"] for c in active_cols], dtype=float)
        b_diff = (b_pred_vec - tgt_vec) / active_scales
        baseline_freq_err.append(abs(float(b_pred_vec[freq_idx] - tgt_vec[freq_idx])))
        baseline_anh_err.append(abs(float(b_pred_vec[anh_idx] - tgt_vec[anh_idx])))
        baseline_norm_err.append(float(np.sqrt(np.mean(np.square(b_diff)))))

        if i % 10 == 0 or i == qn:
            print(f"[{split_name}] processed {i}/{qn}")

    phase3_stats = {
        "freq_mae_GHz": summary_stats(phase3_freq_err),
        "anharm_mae_GHz": summary_stats(phase3_anh_err),
        "normalized_l2": summary_stats(phase3_norm_err),
    }
    baseline_stats = {
        "freq_mae_GHz": summary_stats(baseline_freq_err),
        "anharm_mae_GHz": summary_stats(baseline_anh_err),
        "normalized_l2": summary_stats(baseline_norm_err),
    }

    return {
        "split": split_name,
        "queries": int(qn),
        "phase3": phase3_stats,
        "baseline": baseline_stats,
        "improvement": {
            "freq_mean_delta_GHz": float(baseline_stats["freq_mae_GHz"]["mean"] - phase3_stats["freq_mae_GHz"]["mean"]),
            "anharm_mean_delta_GHz": float(baseline_stats["anharm_mae_GHz"]["mean"] - phase3_stats["anharm_mae_GHz"]["mean"]),
            "normalized_l2_mean_delta": float(baseline_stats["normalized_l2"]["mean"] - phase3_stats["normalized_l2"]["mean"]),
        },
        "runtime_sec": {
            "mean": float(np.mean(elapsed)),
            "p90": float(np.quantile(np.asarray(elapsed), 0.90)),
        },
    }


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

    rng = np.random.default_rng(args.random_state)

    id_report = eval_split("id_test", ctx.id_test_positions, ctx=ctx, args=args, rng=rng)
    ood_report = eval_split("ood", ctx.ood_positions, ctx=ctx, args=args, rng=rng)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_total": int(len(ctx.xg_raw)),
        "rows_train": int(len(ctx.train_positions)),
        "rows_id_test": int(len(ctx.id_test_positions)),
        "rows_ood": int(len(ctx.ood_positions)),
        "config": {
            "max_queries_per_split": int(args.max_queries_per_split),
            "library_split": args.library_split,
            "retrieval_top_k": int(args.retrieval_top_k),
            "population_size": int(args.population_size),
            "iterations": int(args.iterations),
            "elite_fraction": float(args.elite_fraction),
            "explore_fraction": float(args.explore_fraction),
            "start_mutation_scale": float(args.start_mutation_scale),
            "end_mutation_scale": float(args.end_mutation_scale),
            "surrogate_weight": float(args.surrogate_weight),
            "embedding_weight": float(args.embedding_weight),
            "robustness_weight": float(args.robustness_weight),
            "robust_samples": int(args.robust_samples),
            "fabrication_tolerance": float(args.fabrication_tolerance),
            "robust_std_weight": float(args.robust_std_weight),
            "random_state": int(args.random_state),
        },
        "splits": {
            "id_test": id_report,
            "ood": ood_report,
        },
    }

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "phase3_validation_report.json"
    out_md = outdir / "phase3_validation_report.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Phase 3 Validation Report",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        f"Queries per split: {args.max_queries_per_split}",
        "",
        "## ID Test",
        "",
        f"- Phase3 freq MAE mean: {id_report['phase3']['freq_mae_GHz']['mean']:.6f} GHz",
        f"- Baseline freq MAE mean: {id_report['baseline']['freq_mae_GHz']['mean']:.6f} GHz",
        f"- Improvement: {id_report['improvement']['freq_mean_delta_GHz']:.6f} GHz",
        f"- Phase3 normalized L2 mean: {id_report['phase3']['normalized_l2']['mean']:.6f}",
        f"- Baseline normalized L2 mean: {id_report['baseline']['normalized_l2']['mean']:.6f}",
        "",
        "## OOD",
        "",
        f"- Phase3 freq MAE mean: {ood_report['phase3']['freq_mae_GHz']['mean']:.6f} GHz",
        f"- Baseline freq MAE mean: {ood_report['baseline']['freq_mae_GHz']['mean']:.6f} GHz",
        f"- Improvement: {ood_report['improvement']['freq_mean_delta_GHz']:.6f} GHz",
        f"- Phase3 normalized L2 mean: {ood_report['phase3']['normalized_l2']['mean']:.6f}",
        f"- Baseline normalized L2 mean: {ood_report['baseline']['normalized_l2']['mean']:.6f}",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("=== Phase 3 Validation Complete ===")
    print(f"ID freq MAE mean: phase3={id_report['phase3']['freq_mae_GHz']['mean']:.6f}, baseline={id_report['baseline']['freq_mae_GHz']['mean']:.6f}")
    print(f"OOD freq MAE mean: phase3={ood_report['phase3']['freq_mae_GHz']['mean']:.6f}, baseline={ood_report['baseline']['freq_mae_GHz']['mean']:.6f}")
    print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

