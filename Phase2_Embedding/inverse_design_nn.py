#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return F.normalize(self.net(x), p=2, dim=1)


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


def encode_in_batches(model: TwinEncoder, x: np.ndarray, mode: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(dtype=torch.float32)
            zb = model.encode_geom(xb) if mode == "geom" else model.encode_phys(xb)
            out.append(zb.cpu().numpy())
    return l2_normalize(np.vstack(out))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inverse design query with Phase 2 NN embeddings")
    parser.add_argument("--single-csv", type=Path, default=root / "Dataset" / "final_dataset_single.csv")
    parser.add_argument("--bundle-path", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "phase2_nn_bundle.pt")
    parser.add_argument("--phase1-model-path", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib")
    parser.add_argument("--phase1-metadata-path", type=Path, default=root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json")

    parser.add_argument("--freq-01-ghz", type=float, required=True)
    parser.add_argument("--anharmonicity-ghz", type=float, required=True)
    parser.add_argument("--ej-ghz", type=float, default=None)
    parser.add_argument("--ec-ghz", type=float, default=None)
    parser.add_argument("--charge-sensitivity-ghz", type=float, default=None)

    parser.add_argument("--library-split", choices=("train", "all"), default="train")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = torch.load(args.bundle_path, map_location="cpu", weights_only=False)

    geometry_cols: List[str] = list(bundle["geometry_cols"])
    physics_cols: List[str] = list(bundle["physics_cols"])
    row_index_all: List[int] = list(bundle["row_index_all"])

    cfg = bundle["model_config"]
    model = TwinEncoder(
        geom_in=int(cfg["geom_in"]),
        phys_in=int(cfg["phys_in"]),
        hidden_dim=int(cfg["hidden_dim"]),
        emb_dim=int(cfg["emb_dim"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    df_raw = pd.read_csv(args.single_csv)
    req = ["design_id", *geometry_cols, *physics_cols]
    miss = [c for c in req if c not in df_raw.columns]
    if miss:
        raise SystemExit(f"Missing columns in dataset: {miss}")

    base = df_raw[req].copy()
    base["row_index"] = df_raw.index
    base = base.set_index("row_index")
    df = base.loc[row_index_all].reset_index()

    xg_raw = df.loc[:, geometry_cols].to_numpy(dtype=float)
    xp_raw = df.loc[:, physics_cols].to_numpy(dtype=float)

    g_mean = np.array(bundle["geom_scaler_mean"], dtype=np.float32)
    g_scale = np.array(bundle["geom_scaler_scale"], dtype=np.float32)
    p_mean = np.array(bundle["phys_scaler_mean"], dtype=np.float32)
    p_scale = np.array(bundle["phys_scaler_scale"], dtype=np.float32)

    xg_s = ((xg_raw - g_mean) / g_scale).astype(np.float32)
    xp_s = ((xp_raw - p_mean) / p_scale).astype(np.float32)

    g_embed = encode_in_batches(model, xg_s, mode="geom", batch_size=args.batch_size)

    target_map: Dict[str, float] = {
        "freq_01_GHz": float(args.freq_01_ghz),
        "anharmonicity_GHz": float(args.anharmonicity_ghz),
        "EJ_GHz": float(args.ej_ghz) if args.ej_ghz is not None else float(bundle["physics_medians"]["EJ_GHz"]),
        "EC_GHz": float(args.ec_ghz) if args.ec_ghz is not None else float(bundle["physics_medians"]["EC_GHz"]),
        "charge_sensitivity_GHz": (
            float(args.charge_sensitivity_ghz)
            if args.charge_sensitivity_ghz is not None
            else float(bundle["physics_medians"]["charge_sensitivity_GHz"])
        ),
    }
    target_phys = np.array([target_map[c] for c in physics_cols], dtype=np.float32).reshape(1, -1)
    target_phys_s = ((target_phys - p_mean) / p_scale).astype(np.float32)

    with torch.no_grad():
        target_embed = model.encode_phys(torch.from_numpy(target_phys_s)).cpu().numpy()
    target_embed = l2_normalize(target_embed)[0]

    if args.library_split == "train":
        library_pos = np.array(bundle["train_positions"], dtype=int)
    else:
        library_pos = np.arange(len(df), dtype=int)

    sim = g_embed[library_pos] @ target_embed
    k = min(max(1, args.top_k), len(library_pos))
    cand_local = np.argpartition(-sim, kth=k - 1)[:k]
    cand_local = cand_local[np.argsort(-sim[cand_local])]
    cand_pos = library_pos[cand_local]

    rerank_score = np.full(len(cand_pos), np.nan, dtype=float)
    if (not args.no_rerank) and args.phase1_model_path.exists() and args.phase1_metadata_path.exists():
        phase1_model = joblib.load(args.phase1_model_path)
        phase1_meta = json.loads(args.phase1_metadata_path.read_text(encoding="utf-8"))
        phase1_target_cols = list(phase1_meta.get("target_cols", []))
        shared_cols = [c for c in physics_cols if c in phase1_target_cols]
        if shared_cols:
            pidx = [physics_cols.index(c) for c in shared_cols]
            midx = [phase1_target_cols.index(c) for c in shared_cols]
            pscales = np.array([bundle["physics_scales"][c] for c in shared_cols], dtype=float)
            pscales = np.where(pscales <= 1e-12, 1.0, pscales)
            pred = phase1_model.predict(xg_raw[cand_pos])
            pred_shared = pred[:, midx]
            target_shared = target_phys[0, pidx]
            rerank_score = np.sqrt(np.mean(np.square((pred_shared - target_shared) / pscales), axis=1))

    if np.isfinite(rerank_score).any():
        order = np.argsort(rerank_score)
    else:
        order = np.arange(len(cand_pos))

    top_n = min(max(1, args.top_n), len(cand_pos))
    rows = []
    for idx in order[:top_n]:
        pos = int(cand_pos[idx])
        rows.append(
            {
                "rank": int(len(rows) + 1),
                "design_id": int(df.loc[pos, "design_id"]),
                "row_index": int(df.loc[pos, "row_index"]),
                "embed_distance": float(1.0 - (g_embed[pos] @ target_embed)),
                "rerank_score": None if not np.isfinite(rerank_score[idx]) else float(rerank_score[idx]),
                "pad_width_um": float(df.loc[pos, "pad_width_um"]),
                "pad_height_um": float(df.loc[pos, "pad_height_um"]),
                "gap_um": float(df.loc[pos, "gap_um"]),
                "junction_area_um2": float(df.loc[pos, "junction_area_um2"]),
                "freq_01_GHz": float(df.loc[pos, "freq_01_GHz"]),
                "anharmonicity_GHz": float(df.loc[pos, "anharmonicity_GHz"]),
                "EJ_GHz": float(df.loc[pos, "EJ_GHz"]),
                "EC_GHz": float(df.loc[pos, "EC_GHz"]),
                "charge_sensitivity_GHz": float(df.loc[pos, "charge_sensitivity_GHz"]),
            }
        )

    payload = {
        "target": {k: float(v) for k, v in target_map.items()},
        "library_split": args.library_split,
        "top_k": int(k),
        "top_n": int(top_n),
        "rerank_enabled": bool(np.isfinite(rerank_score).any()),
        "results": rows,
    }

    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
