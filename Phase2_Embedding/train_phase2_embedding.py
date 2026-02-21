#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GEOMETRY_COLS: Tuple[str, ...] = (
    "pad_width_um",
    "pad_height_um",
    "gap_um",
    "junction_area_um2",
)

PHYSICS_COLS: Tuple[str, ...] = (
    "freq_01_GHz",
    "anharmonicity_GHz",
    "EJ_GHz",
    "EC_GHz",
    "charge_sensitivity_GHz",
)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def retrieval_metrics(g_embed: np.ndarray, p_embed: np.ndarray) -> Dict[str, float]:
    if g_embed.shape != p_embed.shape:
        raise ValueError("Geometry and physics embeddings must have the same shape")
    n = g_embed.shape[0]
    if n == 0:
        return {
            "n": 0,
            "top1_acc": float("nan"),
            "top5_acc": float("nan"),
            "top10_acc": float("nan"),
            "mrr": float("nan"),
            "pair_cosine_mean": float("nan"),
            "pair_cosine_median": float("nan"),
            "hard_negative_mean": float("nan"),
            "pair_margin_mean": float("nan"),
        }

    sim = g_embed @ p_embed.T
    target = np.arange(n)

    top1 = np.argmax(sim, axis=1)
    top1_acc = float(np.mean(top1 == target))

    def topk_acc(k: int) -> float:
        kk = min(max(1, k), n)
        idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        hits = [(target[i] in idx[i]) for i in range(n)]
        return float(np.mean(hits))

    order = np.argsort(-sim, axis=1)
    rank = np.argmax(order == target[:, None], axis=1)
    mrr = float(np.mean(1.0 / (rank + 1.0)))

    pair = np.diag(sim)
    sim_wo_diag = sim.copy()
    np.fill_diagonal(sim_wo_diag, -np.inf)
    hard_neg = np.max(sim_wo_diag, axis=1)

    return {
        "n": int(n),
        "top1_acc": top1_acc,
        "top5_acc": topk_acc(5),
        "top10_acc": topk_acc(10),
        "mrr": mrr,
        "pair_cosine_mean": float(np.mean(pair)),
        "pair_cosine_median": float(np.median(pair)),
        "hard_negative_mean": float(np.mean(hard_neg)),
        "pair_margin_mean": float(np.mean(pair - hard_neg)),
    }


def canonical_corrs(g_scores: np.ndarray, p_scores: np.ndarray) -> List[float]:
    comps = min(g_scores.shape[1], p_scores.shape[1])
    out: List[float] = []
    for i in range(comps):
        corr = np.corrcoef(g_scores[:, i], p_scores[:, i])[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        out.append(float(corr))
    return out


def build_ood_mask(df: pd.DataFrame, width_q: float, height_q: float, gap_q: float) -> Tuple[np.ndarray, Dict[str, float]]:
    width_thr = float(df["pad_width_um"].quantile(width_q))
    height_thr = float(df["pad_height_um"].quantile(height_q))
    gap_thr = float(df["gap_um"].quantile(gap_q))
    mask = (
        (df["pad_width_um"] >= width_thr)
        | (df["pad_height_um"] >= height_thr)
        | (df["gap_um"] <= gap_thr)
    ).to_numpy()
    thresholds = {
        "pad_width_um_q": width_q,
        "pad_width_um_threshold": width_thr,
        "pad_height_um_q": height_q,
        "pad_height_um_threshold": height_thr,
        "gap_um_q": gap_q,
        "gap_um_threshold": gap_thr,
    }
    return mask, thresholds


def ensure_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Phase 2 shared embeddings")
    parser.add_argument(
        "--single-csv",
        type=Path,
        default=repo_root / "Dataset" / "final_dataset_single.csv",
        help="Path to single-qubit dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Artifact output directory",
    )
    parser.add_argument("--id-test-size", type=float, default=0.2)
    parser.add_argument("--ood-width-quantile", type=float, default=0.85)
    parser.add_argument("--ood-height-quantile", type=float, default=0.85)
    parser.add_argument("--ood-gap-quantile", type=float, default=0.15)
    parser.add_argument("--rbf-components", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--rbf-gamma", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.id_test_size <= 0 or args.id_test_size >= 1:
        raise SystemExit("id-test-size must be in (0, 1)")

    df_raw = pd.read_csv(args.single_csv)
    ensure_required_columns(df_raw, GEOMETRY_COLS + PHYSICS_COLS + ("design_id",))

    selected_cols = ["design_id", *GEOMETRY_COLS, *PHYSICS_COLS]
    df = df_raw[selected_cols].copy()
    df["row_index"] = df_raw.index.to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if df.empty:
        raise SystemExit("No valid rows left after cleaning")

    ood_mask, thresholds = build_ood_mask(
        df=df,
        width_q=args.ood_width_quantile,
        height_q=args.ood_height_quantile,
        gap_q=args.ood_gap_quantile,
    )

    all_pos = np.arange(len(df))
    in_dist_pos = all_pos[~ood_mask]
    ood_pos = all_pos[ood_mask]
    if len(in_dist_pos) < 100:
        raise SystemExit("Too few in-distribution rows to train")
    if len(ood_pos) < 50:
        rng = np.random.default_rng(args.random_state)
        ood_pos = rng.choice(all_pos, size=max(50, int(0.15 * len(all_pos))), replace=False)
        ood_mask = np.zeros(len(all_pos), dtype=bool)
        ood_mask[ood_pos] = True
        in_dist_pos = all_pos[~ood_mask]

    train_pos, id_test_pos = train_test_split(
        in_dist_pos,
        test_size=args.id_test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    xg = df.loc[:, GEOMETRY_COLS].to_numpy(dtype=float)
    xp = df.loc[:, PHYSICS_COLS].to_numpy(dtype=float)

    xg_train = xg[train_pos]
    xp_train = xp[train_pos]
    xg_id = xg[id_test_pos]
    xp_id = xp[id_test_pos]
    xg_ood = xg[ood_pos]
    xp_ood = xp[ood_pos]

    g_scaler = StandardScaler().fit(xg_train)
    p_scaler = StandardScaler().fit(xp_train)

    xg_train_s = g_scaler.transform(xg_train)
    xp_train_s = p_scaler.transform(xp_train)
    xg_id_s = g_scaler.transform(xg_id)
    xp_id_s = p_scaler.transform(xp_id)
    xg_ood_s = g_scaler.transform(xg_ood)
    xp_ood_s = p_scaler.transform(xp_ood)

    g_mapper = RBFSampler(
        gamma=args.rbf_gamma,
        n_components=args.rbf_components,
        random_state=args.random_state,
    )
    p_mapper = RBFSampler(
        gamma=args.rbf_gamma,
        n_components=args.rbf_components,
        random_state=args.random_state + 1,
    )

    xg_train_r = g_mapper.fit_transform(xg_train_s)
    xp_train_r = p_mapper.fit_transform(xp_train_s)
    xg_id_r = g_mapper.transform(xg_id_s)
    xp_id_r = p_mapper.transform(xp_id_s)
    xg_ood_r = g_mapper.transform(xg_ood_s)
    xp_ood_r = p_mapper.transform(xp_ood_s)

    max_components = min(
        args.embedding_dim,
        args.rbf_components,
        xg_train_r.shape[0] - 1,
        xp_train_r.shape[0] - 1,
    )
    if max_components < 2:
        raise SystemExit("Not enough training rows to fit embedding")

    cca = CCA(n_components=max_components, max_iter=1000, tol=1e-6)
    cca.fit(xg_train_r, xp_train_r)

    g_train_c, p_train_c = cca.transform(xg_train_r, xp_train_r)
    g_id_c, p_id_c = cca.transform(xg_id_r, xp_id_r)
    g_ood_c, p_ood_c = cca.transform(xg_ood_r, xp_ood_r)

    g_train_e = l2_normalize(g_train_c)
    p_train_e = l2_normalize(p_train_c)
    g_id_e = l2_normalize(g_id_c)
    p_id_e = l2_normalize(p_id_c)
    g_ood_e = l2_normalize(g_ood_c)
    p_ood_e = l2_normalize(p_ood_c)

    id_metrics = retrieval_metrics(g_id_e, p_id_e)
    ood_metrics = retrieval_metrics(g_ood_e, p_ood_e)

    xg_all_r = g_mapper.transform(g_scaler.transform(xg))
    xp_all_r = p_mapper.transform(p_scaler.transform(xp))
    g_all_c, p_all_c = cca.transform(xg_all_r, xp_all_r)
    g_all_e = l2_normalize(g_all_c)
    p_all_e = l2_normalize(p_all_c)

    split = np.array(["train"] * len(df), dtype=object)
    split[id_test_pos] = "id_test"
    split[ood_pos] = "ood"

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    embed_cols_g = [f"geom_emb_{i}" for i in range(g_all_e.shape[1])]
    embed_cols_p = [f"phys_emb_{i}" for i in range(p_all_e.shape[1])]
    emb_df = pd.DataFrame(
        {
            "row_index": df["row_index"],
            "design_id": df["design_id"],
            "split": split,
            **{embed_cols_g[i]: g_all_e[:, i] for i in range(g_all_e.shape[1])},
            **{embed_cols_p[i]: p_all_e[:, i] for i in range(p_all_e.shape[1])},
        }
    )
    emb_df.to_csv(outdir / "phase2_all_embeddings.csv", index=False)

    split_df = pd.DataFrame(
        {
            "row_index": df["row_index"],
            "design_id": df["design_id"],
            "split": split,
        }
    )
    split_df.to_csv(outdir / "phase2_splits.csv", index=False)

    phys_scale = np.std(xp_train, axis=0)
    phys_scale = np.where(phys_scale <= 1e-12, 1.0, phys_scale)

    bundle = {
        "version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.single_csv.resolve()),
        "geometry_cols": list(GEOMETRY_COLS),
        "physics_cols": list(PHYSICS_COLS),
        "row_index_all": df["row_index"].astype(int).tolist(),
        "design_id_all": df["design_id"].astype(int).tolist(),
        "train_positions": train_pos.astype(int).tolist(),
        "id_test_positions": id_test_pos.astype(int).tolist(),
        "ood_positions": ood_pos.astype(int).tolist(),
        "g_scaler": g_scaler,
        "p_scaler": p_scaler,
        "g_mapper": g_mapper,
        "p_mapper": p_mapper,
        "cca": cca,
        "embedding_dim": int(max_components),
        "rbf_components": int(args.rbf_components),
        "rbf_gamma": float(args.rbf_gamma),
        "random_state": int(args.random_state),
        "thresholds": thresholds,
        "physics_medians": {
            col: float(np.median(xp_train[:, i])) for i, col in enumerate(PHYSICS_COLS)
        },
        "physics_scales": {
            col: float(phys_scale[i]) for i, col in enumerate(PHYSICS_COLS)
        },
    }
    joblib.dump(bundle, outdir / "phase2_embedding_bundle.joblib")

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_pos)),
        "rows_id_test": int(len(id_test_pos)),
        "rows_ood": int(len(ood_pos)),
        "embedding_dim": int(max_components),
        "canonical_correlation_train": canonical_corrs(g_train_c, p_train_c),
        "metrics_id": id_metrics,
        "metrics_ood": ood_metrics,
        "thresholds": thresholds,
        "trained_at_utc": bundle["trained_at_utc"],
    }

    (outdir / "phase2_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Phase 2 Embedding Trained ===")
    print(f"Rows total={summary['rows_total']} train={summary['rows_train']} id_test={summary['rows_id_test']} ood={summary['rows_ood']}")
    print(f"Embedding dim={summary['embedding_dim']}")
    print(f"ID retrieval top1={id_metrics['top1_acc']:.4f} top5={id_metrics['top5_acc']:.4f}")
    print(f"OOD retrieval top1={ood_metrics['top1_acc']:.4f} top5={ood_metrics['top5_acc']:.4f}")
    print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
