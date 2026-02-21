#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


class EncoderMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)


class TwinEncoder(nn.Module):
    def __init__(self, geom_in: int, phys_in: int, hidden_dim: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.geom_encoder = EncoderMLP(geom_in, hidden_dim, emb_dim, dropout)
        self.phys_encoder = EncoderMLP(phys_in, hidden_dim, emb_dim, dropout)

    def encode_geom(self, x: torch.Tensor) -> torch.Tensor:
        return self.geom_encoder(x)

    def encode_phys(self, x: torch.Tensor) -> torch.Tensor:
        return self.phys_encoder(x)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    return x / n


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


def nt_xent_loss(zg: torch.Tensor, zp: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (zg @ zp.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_gp = F.cross_entropy(logits, labels)
    loss_pg = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_gp + loss_pg)


def ensure_required_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


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


def encode_in_batches(model: TwinEncoder, x: np.ndarray, mode: str, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device=device, dtype=torch.float32)
            zb = model.encode_geom(xb) if mode == "geom" else model.encode_phys(xb)
            out.append(zb.cpu().numpy())
    return l2_normalize(np.vstack(out))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Phase 2 NN twin-encoder embeddings")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    parser.add_argument("--id-test-size", type=float, default=0.2)
    parser.add_argument("--ood-width-quantile", type=float, default=0.95)
    parser.add_argument("--ood-height-quantile", type=float, default=0.95)
    parser.add_argument("--ood-gap-quantile", type=float, default=0.05)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    df_raw = pd.read_csv(args.single_csv)
    ensure_required_columns(df_raw, GEOMETRY_COLS + PHYSICS_COLS + ("design_id",))

    selected_cols = ["design_id", *GEOMETRY_COLS, *PHYSICS_COLS]
    df = df_raw[selected_cols].copy()
    df["row_index"] = df_raw.index.to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(df) < 500:
        raise SystemExit("Not enough rows after cleaning")

    ood_mask, thresholds = build_ood_mask(
        df,
        width_q=args.ood_width_quantile,
        height_q=args.ood_height_quantile,
        gap_q=args.ood_gap_quantile,
    )
    all_pos = np.arange(len(df))
    in_pos = all_pos[~ood_mask]
    ood_pos = all_pos[ood_mask]
    if len(in_pos) < 200 or len(ood_pos) < 100:
        raise SystemExit("Split too small; adjust OOD quantiles")

    train_pos, id_test_pos = train_test_split(
        in_pos,
        test_size=args.id_test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    xg = df.loc[:, GEOMETRY_COLS].to_numpy(dtype=float)
    xp = df.loc[:, PHYSICS_COLS].to_numpy(dtype=float)

    g_scaler = StandardScaler().fit(xg[train_pos])
    p_scaler = StandardScaler().fit(xp[train_pos])

    g_mean = g_scaler.mean_.astype(np.float32)
    g_scale = np.where(g_scaler.scale_ <= 1e-12, 1.0, g_scaler.scale_).astype(np.float32)
    p_mean = p_scaler.mean_.astype(np.float32)
    p_scale = np.where(p_scaler.scale_ <= 1e-12, 1.0, p_scaler.scale_).astype(np.float32)

    xg_s = ((xg - g_mean) / g_scale).astype(np.float32)
    xp_s = ((xp - p_mean) / p_scale).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(xg_s[train_pos]),
        torch.from_numpy(xp_s[train_pos]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cpu")
    model = TwinEncoder(
        geom_in=len(GEOMETRY_COLS),
        phys_in=len(PHYSICS_COLS),
        hidden_dim=args.hidden_dim,
        emb_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for g_batch, p_batch in train_loader:
            g_batch = g_batch.to(device)
            p_batch = p_batch.to(device)

            zg = model.encode_geom(g_batch)
            zp = model.encode_phys(p_batch)
            loss = nt_xent_loss(zg, zp, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        do_eval = (epoch == 1) or (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if do_eval:
            g_id = encode_in_batches(model, xg_s[id_test_pos], mode="geom", batch_size=args.eval_batch_size, device=device)
            p_id = encode_in_batches(model, xp_s[id_test_pos], mode="phys", batch_size=args.eval_batch_size, device=device)
            id_metrics = retrieval_metrics(g_id, p_id)
            score = float(id_metrics["top5_acc"] + 0.5 * id_metrics["top1_acc"])

            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "id_top1": float(id_metrics["top1_acc"]),
                "id_top5": float(id_metrics["top5_acc"]),
                "id_top10": float(id_metrics["top10_acc"]),
                "id_mrr": float(id_metrics["mrr"]),
                "score": score,
            }
            history.append(row)
            print(
                f"epoch={epoch:03d} loss={train_loss:.5f} "
                f"id_top1={row['id_top1']:.4f} id_top5={row['id_top5']:.4f}"
            )

            if score > best_score + 1e-6:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += args.eval_every

            if epochs_no_improve >= args.patience:
                print(f"early_stop at epoch={epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    g_all = encode_in_batches(model, xg_s, mode="geom", batch_size=args.eval_batch_size, device=device)
    p_all = encode_in_batches(model, xp_s, mode="phys", batch_size=args.eval_batch_size, device=device)
    g_id = g_all[id_test_pos]
    p_id = p_all[id_test_pos]
    g_ood = g_all[ood_pos]
    p_ood = p_all[ood_pos]

    id_metrics = retrieval_metrics(g_id, p_id)
    ood_metrics = retrieval_metrics(g_ood, p_ood)

    split = np.array(["train"] * len(df), dtype=object)
    split[id_test_pos] = "id_test"
    split[ood_pos] = "ood"

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    emb_g_cols = [f"geom_emb_{i}" for i in range(g_all.shape[1])]
    emb_p_cols = [f"phys_emb_{i}" for i in range(p_all.shape[1])]
    emb_df = pd.DataFrame(
        {
            "row_index": df["row_index"],
            "design_id": df["design_id"],
            "split": split,
            **{emb_g_cols[i]: g_all[:, i] for i in range(g_all.shape[1])},
            **{emb_p_cols[i]: p_all[:, i] for i in range(p_all.shape[1])},
        }
    )
    emb_df.to_csv(outdir / "phase2_nn_all_embeddings.csv", index=False)

    pd.DataFrame(
        {
            "row_index": df["row_index"],
            "design_id": df["design_id"],
            "split": split,
        }
    ).to_csv(outdir / "phase2_nn_splits.csv", index=False)

    physics_scale_train = np.std(xp[train_pos], axis=0)
    physics_scale_train = np.where(physics_scale_train <= 1e-12, 1.0, physics_scale_train)

    bundle = {
        "version": 1,
        "type": "phase2_nn",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.single_csv.resolve()),
        "geometry_cols": list(GEOMETRY_COLS),
        "physics_cols": list(PHYSICS_COLS),
        "row_index_all": df["row_index"].astype(int).tolist(),
        "design_id_all": df["design_id"].astype(int).tolist(),
        "train_positions": train_pos.astype(int).tolist(),
        "id_test_positions": id_test_pos.astype(int).tolist(),
        "ood_positions": ood_pos.astype(int).tolist(),
        "geom_scaler_mean": g_mean.tolist(),
        "geom_scaler_scale": g_scale.tolist(),
        "phys_scaler_mean": p_mean.tolist(),
        "phys_scaler_scale": p_scale.tolist(),
        "model_config": {
            "geom_in": len(GEOMETRY_COLS),
            "phys_in": len(PHYSICS_COLS),
            "hidden_dim": int(args.hidden_dim),
            "emb_dim": int(args.embedding_dim),
            "dropout": float(args.dropout),
        },
        "state_dict": model.state_dict(),
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "random_state": int(args.random_state),
        },
        "best_epoch": int(best_epoch),
        "thresholds": thresholds,
        "physics_medians": {col: float(np.median(xp[train_pos, i])) for i, col in enumerate(PHYSICS_COLS)},
        "physics_scales": {col: float(physics_scale_train[i]) for i, col in enumerate(PHYSICS_COLS)},
    }
    torch.save(bundle, outdir / "phase2_nn_bundle.pt")

    summary = {
        "rows_total": int(len(df)),
        "rows_train": int(len(train_pos)),
        "rows_id_test": int(len(id_test_pos)),
        "rows_ood": int(len(ood_pos)),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "metrics_id": id_metrics,
        "metrics_ood": ood_metrics,
        "thresholds": thresholds,
        "history": history[-60:],
        "trained_at_utc": bundle["trained_at_utc"],
    }
    (outdir / "phase2_nn_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Phase 2 NN Embedding Trained ===")
    print(f"rows total={summary['rows_total']} train={summary['rows_train']} id_test={summary['rows_id_test']} ood={summary['rows_ood']}")
    print(f"best_epoch={summary['best_epoch']} score={summary['best_score']:.4f}")
    print(f"ID retrieval top1={id_metrics['top1_acc']:.4f} top5={id_metrics['top5_acc']:.4f}")
    print(f"OOD retrieval top1={ood_metrics['top1_acc']:.4f} top5={ood_metrics['top5_acc']:.4f}")
    print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
