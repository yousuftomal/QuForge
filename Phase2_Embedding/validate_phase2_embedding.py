#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def retrieval_metrics(g_embed: np.ndarray, p_embed: np.ndarray) -> Dict[str, float]:
    if g_embed.shape != p_embed.shape:
        raise ValueError("Shape mismatch for retrieval metrics")
    n = g_embed.shape[0]
    if n == 0:
        return {
            "n": 0,
            "top1_acc": float("nan"),
            "top5_acc": float("nan"),
            "top10_acc": float("nan"),
            "mrr": float("nan"),
            "pair_cosine_mean": float("nan"),
            "pair_margin_mean": float("nan"),
        }

    sim = g_embed @ p_embed.T
    target = np.arange(n)
    top1 = np.argmax(sim, axis=1)

    def topk_acc(k: int) -> float:
        kk = min(max(1, k), n)
        idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        return float(np.mean([(target[i] in idx[i]) for i in range(n)]))

    order = np.argsort(-sim, axis=1)
    rank = np.argmax(order == target[:, None], axis=1)

    pos = np.diag(sim)
    sim_wo = sim.copy()
    np.fill_diagonal(sim_wo, -np.inf)
    hard_neg = np.max(sim_wo, axis=1)

    return {
        "n": int(n),
        "top1_acc": float(np.mean(top1 == target)),
        "top5_acc": topk_acc(5),
        "top10_acc": topk_acc(10),
        "mrr": float(np.mean(1.0 / (rank + 1.0))),
        "pair_cosine_mean": float(np.mean(pos)),
        "pair_margin_mean": float(np.mean(pos - hard_neg)),
    }


def transform_embeddings(bundle: Dict[str, object], xg: np.ndarray, xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g_scaler = bundle["g_scaler"]
    p_scaler = bundle["p_scaler"]
    g_mapper = bundle["g_mapper"]
    p_mapper = bundle["p_mapper"]
    cca = bundle["cca"]

    xg_r = g_mapper.transform(g_scaler.transform(xg))
    xp_r = p_mapper.transform(p_scaler.transform(xp))
    g_c, p_c = cca.transform(xg_r, xp_r)
    return l2_normalize(g_c), l2_normalize(p_c)


def inverse_eval(
    xg: np.ndarray,
    xp: np.ndarray,
    g_embed: np.ndarray,
    p_embed: np.ndarray,
    physics_cols: Sequence[str],
    physics_scales: np.ndarray,
    library_positions: np.ndarray,
    query_positions: np.ndarray,
    max_queries: int,
    top_k: int,
    random_state: int,
    phase1_model: Optional[object],
    phase1_target_cols: Sequence[str],
) -> Dict[str, object]:
    rng = np.random.default_rng(random_state)
    queries = query_positions.copy()
    if len(queries) > max_queries:
        queries = rng.choice(queries, size=max_queries, replace=False)

    freq_idx = physics_cols.index("freq_01_GHz")
    anh_idx = physics_cols.index("anharmonicity_GHz")

    shared_cols = [c for c in physics_cols if c in phase1_target_cols]
    shared_phys_idx = [physics_cols.index(c) for c in shared_cols]
    shared_phase_idx = [phase1_target_cols.index(c) for c in shared_cols]
    shared_scales = np.array([physics_scales[i] for i in shared_phys_idx], dtype=float)

    emb_freq_abs: List[float] = []
    emb_anh_abs: List[float] = []
    emb_norm_l2: List[float] = []

    rr_freq_abs: List[float] = []
    rr_anh_abs: List[float] = []
    rr_norm_l2: List[float] = []

    for q in queries:
        target = xp[q]
        sim = g_embed[library_positions] @ p_embed[q]
        k = min(max(1, top_k), len(library_positions))
        cand_local = np.argpartition(-sim, kth=k - 1)[:k]
        cand_local = cand_local[np.argsort(-sim[cand_local])]
        cand_global = library_positions[cand_local]

        emb_choice = int(cand_global[0])
        emb_vec = xp[emb_choice]
        emb_diff = (emb_vec - target) / physics_scales
        emb_freq_abs.append(abs(float(emb_vec[freq_idx] - target[freq_idx])))
        emb_anh_abs.append(abs(float(emb_vec[anh_idx] - target[anh_idx])))
        emb_norm_l2.append(float(np.sqrt(np.mean(np.square(emb_diff)))))

        rr_choice = emb_choice
        if phase1_model is not None and shared_cols:
            pred = phase1_model.predict(xg[cand_global])
            pred_shared = pred[:, shared_phase_idx]
            target_shared = target[shared_phys_idx]
            rr_score = np.sqrt(np.mean(np.square((pred_shared - target_shared) / shared_scales), axis=1))
            rr_choice = int(cand_global[int(np.argmin(rr_score))])

        rr_vec = xp[rr_choice]
        rr_diff = (rr_vec - target) / physics_scales
        rr_freq_abs.append(abs(float(rr_vec[freq_idx] - target[freq_idx])))
        rr_anh_abs.append(abs(float(rr_vec[anh_idx] - target[anh_idx])))
        rr_norm_l2.append(float(np.sqrt(np.mean(np.square(rr_diff)))))

    emb_mean = {
        "freq_mae": float(np.mean(emb_freq_abs)),
        "anharm_mae": float(np.mean(emb_anh_abs)),
        "normalized_l2": float(np.mean(emb_norm_l2)),
    }
    rr_mean = {
        "freq_mae": float(np.mean(rr_freq_abs)),
        "anharm_mae": float(np.mean(rr_anh_abs)),
        "normalized_l2": float(np.mean(rr_norm_l2)),
    }

    return {
        "query_count": int(len(queries)),
        "embedding_only": emb_mean,
        "reranked": rr_mean,
        "improvement": {
            "freq_mae_delta": float(emb_mean["freq_mae"] - rr_mean["freq_mae"]),
            "anharm_mae_delta": float(emb_mean["anharm_mae"] - rr_mean["anharm_mae"]),
            "normalized_l2_delta": float(emb_mean["normalized_l2"] - rr_mean["normalized_l2"]),
        },
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Phase 2 embedding and inverse-design performance")
    parser.add_argument(
        "--single-csv",
        type=Path,
        default=repo_root / "Dataset" / "final_dataset_single.csv",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "phase2_embedding_bundle.joblib",
    )
    parser.add_argument(
        "--phase1-model-path",
        type=Path,
        default=repo_root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib",
    )
    parser.add_argument(
        "--phase1-metadata-path",
        type=Path,
        default=repo_root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-queries", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)

    geometry_cols: List[str] = list(bundle["geometry_cols"])
    physics_cols: List[str] = list(bundle["physics_cols"])
    row_index_all: List[int] = list(bundle["row_index_all"])

    df_raw = pd.read_csv(args.single_csv)
    required_cols = ["design_id", *geometry_cols, *physics_cols]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns in dataset: {missing}")

    base = df_raw[required_cols].copy()
    base["row_index"] = df_raw.index
    base = base.set_index("row_index")
    try:
        df = base.loc[row_index_all].reset_index()
    except KeyError as exc:
        raise SystemExit("Dataset row mapping does not match training bundle") from exc

    xg = df.loc[:, geometry_cols].to_numpy(dtype=float)
    xp = df.loc[:, physics_cols].to_numpy(dtype=float)

    g_embed, p_embed = transform_embeddings(bundle, xg, xp)

    train_pos = np.array(bundle["train_positions"], dtype=int)
    id_test_pos = np.array(bundle["id_test_positions"], dtype=int)
    ood_pos = np.array(bundle["ood_positions"], dtype=int)

    id_metrics = retrieval_metrics(g_embed[id_test_pos], p_embed[id_test_pos])
    ood_metrics = retrieval_metrics(g_embed[ood_pos], p_embed[ood_pos])

    phys_scale = np.array([bundle["physics_scales"][c] for c in physics_cols], dtype=float)
    phys_scale = np.where(phys_scale <= 1e-12, 1.0, phys_scale)

    phase1_model = None
    phase1_target_cols: List[str] = []
    if args.phase1_model_path.exists() and args.phase1_metadata_path.exists():
        phase1_model = joblib.load(args.phase1_model_path)
        phase1_meta = json.loads(args.phase1_metadata_path.read_text(encoding="utf-8"))
        phase1_target_cols = list(phase1_meta.get("target_cols", []))

    id_inverse = inverse_eval(
        xg=xg,
        xp=xp,
        g_embed=g_embed,
        p_embed=p_embed,
        physics_cols=physics_cols,
        physics_scales=phys_scale,
        library_positions=train_pos,
        query_positions=id_test_pos,
        max_queries=args.max_queries,
        top_k=args.top_k,
        random_state=args.random_state,
        phase1_model=phase1_model,
        phase1_target_cols=phase1_target_cols,
    )

    ood_inverse = inverse_eval(
        xg=xg,
        xp=xp,
        g_embed=g_embed,
        p_embed=p_embed,
        physics_cols=physics_cols,
        physics_scales=phys_scale,
        library_positions=train_pos,
        query_positions=ood_pos,
        max_queries=args.max_queries,
        top_k=args.top_k,
        random_state=args.random_state + 7,
        phase1_model=phase1_model,
        phase1_target_cols=phase1_target_cols,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(Path(args.bundle_path).resolve()),
        "single_csv": str(Path(args.single_csv).resolve()),
        "rows": {
            "total": int(len(df)),
            "train": int(len(train_pos)),
            "id_test": int(len(id_test_pos)),
            "ood": int(len(ood_pos)),
        },
        "retrieval": {
            "id": id_metrics,
            "ood": ood_metrics,
        },
        "inverse_design": {
            "id": id_inverse,
            "ood": ood_inverse,
            "phase1_rerank_enabled": bool(phase1_model is not None and len(phase1_target_cols) > 0),
        },
    }

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "phase2_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = []
    md.append("# Phase 2 Validation Report")
    md.append("")
    md.append(f"Generated: {report['generated_at_utc']}")
    md.append("")
    md.append("## Retrieval")
    md.append("")
    md.append(f"- ID top1: {id_metrics['top1_acc']:.4f}, top5: {id_metrics['top5_acc']:.4f}, margin: {id_metrics['pair_margin_mean']:.4f}")
    md.append(f"- OOD top1: {ood_metrics['top1_acc']:.4f}, top5: {ood_metrics['top5_acc']:.4f}, margin: {ood_metrics['pair_margin_mean']:.4f}")
    md.append("")
    md.append("## Inverse Design")
    md.append("")
    md.append(f"- ID embedding-only freq MAE: {id_inverse['embedding_only']['freq_mae']:.6f}")
    md.append(f"- ID reranked freq MAE: {id_inverse['reranked']['freq_mae']:.6f}")
    md.append(f"- OOD embedding-only freq MAE: {ood_inverse['embedding_only']['freq_mae']:.6f}")
    md.append(f"- OOD reranked freq MAE: {ood_inverse['reranked']['freq_mae']:.6f}")
    md.append("")
    md.append("## Artifacts")
    md.append("")
    md.append("- `phase2_validation_report.json`")
    md.append("- `phase2_validation_report.md`")

    (outdir / "phase2_validation_report.md").write_text("\n".join(md), encoding="utf-8")

    print("=== Phase 2 Validation Complete ===")
    print(f"ID retrieval top1={id_metrics['top1_acc']:.4f} top5={id_metrics['top5_acc']:.4f}")
    print(f"OOD retrieval top1={ood_metrics['top1_acc']:.4f} top5={ood_metrics['top5_acc']:.4f}")
    print(
        "ID inverse freq MAE: "
        f"embed={id_inverse['embedding_only']['freq_mae']:.6f}, "
        f"rerank={id_inverse['reranked']['freq_mae']:.6f}"
    )
    print(
        "OOD inverse freq MAE: "
        f"embed={ood_inverse['embedding_only']['freq_mae']:.6f}, "
        f"rerank={ood_inverse['reranked']['freq_mae']:.6f}"
    )
    print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
