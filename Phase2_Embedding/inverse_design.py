#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inverse-design query using Phase 2 embeddings")
    parser.add_argument("--single-csv", type=Path, default=repo_root / "Dataset" / "final_dataset_single.csv")
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

    parser.add_argument("--freq-01-ghz", type=float, required=True)
    parser.add_argument("--anharmonicity-ghz", type=float, required=True)
    parser.add_argument("--ej-ghz", type=float, default=None)
    parser.add_argument("--ec-ghz", type=float, default=None)
    parser.add_argument("--charge-sensitivity-ghz", type=float, default=None)

    parser.add_argument("--library-split", choices=("train", "all"), default="train")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)

    geometry_cols: List[str] = list(bundle["geometry_cols"])
    physics_cols: List[str] = list(bundle["physics_cols"])
    row_index_all: List[int] = list(bundle["row_index_all"])
    physics_medians: Dict[str, float] = dict(bundle["physics_medians"])
    physics_scales = np.array([bundle["physics_scales"][c] for c in physics_cols], dtype=float)
    physics_scales = np.where(physics_scales <= 1e-12, 1.0, physics_scales)

    df_raw = pd.read_csv(args.single_csv)
    required = ["design_id", *geometry_cols, *physics_cols]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}")

    base = df_raw[required].copy()
    base["row_index"] = df_raw.index
    base = base.set_index("row_index")
    df = base.loc[row_index_all].reset_index()

    xg = df.loc[:, geometry_cols].to_numpy(dtype=float)
    xp = df.loc[:, physics_cols].to_numpy(dtype=float)

    g_scaler = bundle["g_scaler"]
    p_scaler = bundle["p_scaler"]
    g_mapper = bundle["g_mapper"]
    p_mapper = bundle["p_mapper"]
    cca = bundle["cca"]

    xg_r = g_mapper.transform(g_scaler.transform(xg))
    xp_r = p_mapper.transform(p_scaler.transform(xp))
    g_c, _ = cca.transform(xg_r, xp_r)
    g_embed = l2_normalize(g_c)

    target_map: Dict[str, float] = {
        "freq_01_GHz": float(args.freq_01_ghz),
        "anharmonicity_GHz": float(args.anharmonicity_ghz),
        "EJ_GHz": float(args.ej_ghz) if args.ej_ghz is not None else float(physics_medians["EJ_GHz"]),
        "EC_GHz": float(args.ec_ghz) if args.ec_ghz is not None else float(physics_medians["EC_GHz"]),
        "charge_sensitivity_GHz": (
            float(args.charge_sensitivity_ghz)
            if args.charge_sensitivity_ghz is not None
            else float(physics_medians["charge_sensitivity_GHz"])
        ),
    }
    target_phys = np.array([target_map[c] for c in physics_cols], dtype=float).reshape(1, -1)

    dummy_geom = np.zeros((1, len(geometry_cols)), dtype=float)
    dummy_r = g_mapper.transform(g_scaler.transform(dummy_geom))
    target_r = p_mapper.transform(p_scaler.transform(target_phys))
    _, target_c = cca.transform(dummy_r, target_r)
    target_embed = l2_normalize(target_c)[0]

    if args.library_split == "train":
        library_pos = np.array(bundle["train_positions"], dtype=int)
    else:
        library_pos = np.arange(len(df), dtype=int)

    sim = g_embed[library_pos] @ target_embed
    k = min(max(1, args.top_k), len(library_pos))
    cand_local = np.argpartition(-sim, kth=k - 1)[:k]
    cand_local = cand_local[np.argsort(-sim[cand_local])]
    cand_pos = library_pos[cand_local]

    phase1_model = None
    phase1_target_cols: List[str] = []
    if (not args.no_rerank) and args.phase1_model_path.exists() and args.phase1_metadata_path.exists():
        phase1_model = joblib.load(args.phase1_model_path)
        phase1_meta = json.loads(args.phase1_metadata_path.read_text(encoding="utf-8"))
        phase1_target_cols = list(phase1_meta.get("target_cols", []))

    score_embed = np.array([float(1.0 - sim[np.where(cand_local == i)[0][0]]) for i in cand_local])

    rerank_score = np.full(len(cand_pos), np.nan, dtype=float)
    if phase1_model is not None:
        shared_cols = [c for c in physics_cols if c in phase1_target_cols]
        if shared_cols:
            phys_idx = [physics_cols.index(c) for c in shared_cols]
            phase_idx = [phase1_target_cols.index(c) for c in shared_cols]
            shared_scale = np.array([physics_scales[i] for i in phys_idx], dtype=float)
            pred = phase1_model.predict(xg[cand_pos])
            pred_shared = pred[:, phase_idx]
            target_shared = target_phys[0, phys_idx]
            rerank_score = np.sqrt(np.mean(np.square((pred_shared - target_shared) / shared_scale), axis=1))

    if np.isfinite(rerank_score).any():
        final_order = np.argsort(rerank_score)
    else:
        final_order = np.arange(len(cand_pos))

    top_n = min(max(1, args.top_n), len(cand_pos))
    rows = []
    for rank_idx in final_order[:top_n]:
        pos = int(cand_pos[rank_idx])
        rec = {
            "rank": int(len(rows) + 1),
            "design_id": int(df.loc[pos, "design_id"]),
            "row_index": int(df.loc[pos, "row_index"]),
            "embed_distance": float(1.0 - (g_embed[pos] @ target_embed)),
            "rerank_score": None if not np.isfinite(rerank_score[rank_idx]) else float(rerank_score[rank_idx]),
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
        rows.append(rec)

    payload = {
        "target": target_map,
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
