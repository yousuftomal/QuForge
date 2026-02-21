#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    n = np.where(n <= 1e-12, 1.0, n)
    return x / n


@dataclass(frozen=True)
class Phase3Context:
    geometry_cols: Tuple[str, ...]
    physics_cols: Tuple[str, ...]
    physics_scales: np.ndarray
    physics_medians: Dict[str, float]
    row_index_all: np.ndarray
    design_id_all: np.ndarray
    xg_raw: np.ndarray
    xp_raw: np.ndarray
    g_mean: np.ndarray
    g_scale: np.ndarray
    p_mean: np.ndarray
    p_scale: np.ndarray
    g_embed: np.ndarray
    train_positions: np.ndarray
    id_test_positions: np.ndarray
    ood_positions: np.ndarray
    model: TwinEncoder
    phase1_model: Any
    phase1_feature_cols: Tuple[str, ...]
    phase1_target_cols: Tuple[str, ...]
    phase1_feature_indices: np.ndarray
    shared_target_cols: Tuple[str, ...]
    shared_phys_indices: np.ndarray
    shared_phase1_indices: np.ndarray
    shared_scales: np.ndarray


def default_paths(root: Path) -> Dict[str, Path]:
    return {
        "single_csv": root / "Dataset" / "final_dataset_single.csv",
        "phase2_bundle": root / "Phase2_Embedding" / "artifacts" / "phase2_nn_bundle.pt",
        "phase1_model": root / "Phase1_Surrogate" / "artifacts" / "single_surrogate.joblib",
        "phase1_meta": root / "Phase1_Surrogate" / "artifacts" / "single_metadata.json",
    }


def _encode_geom(model: TwinEncoder, x_scaled: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_scaled.shape[0], batch_size):
            xb = torch.from_numpy(x_scaled[start : start + batch_size]).to(dtype=torch.float32)
            zb = model.encode_geom(xb).cpu().numpy()
            out.append(zb)
    return l2_normalize(np.vstack(out))


def _encode_phys(model: TwinEncoder, x_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model.encode_phys(torch.from_numpy(x_scaled).to(dtype=torch.float32)).cpu().numpy()
    return l2_normalize(z)


def load_context(
    *,
    root: Path,
    single_csv: Optional[Path] = None,
    phase2_bundle: Optional[Path] = None,
    phase1_model_path: Optional[Path] = None,
    phase1_meta_path: Optional[Path] = None,
) -> Phase3Context:
    paths = default_paths(root)
    single_csv = single_csv or paths["single_csv"]
    phase2_bundle = phase2_bundle or paths["phase2_bundle"]
    phase1_model_path = phase1_model_path or paths["phase1_model"]
    phase1_meta_path = phase1_meta_path or paths["phase1_meta"]

    bundle = torch.load(phase2_bundle, map_location="cpu", weights_only=False)

    geometry_cols = tuple(bundle["geometry_cols"])
    physics_cols = tuple(bundle["physics_cols"])
    row_index_all = np.asarray(bundle["row_index_all"], dtype=int)
    design_id_all = np.asarray(bundle["design_id_all"], dtype=int)

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

    df_raw = pd.read_csv(single_csv)
    req = ["design_id", *geometry_cols, *physics_cols]
    missing = [c for c in req if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    base = df_raw[req].copy()
    base["row_index"] = df_raw.index
    base = base.set_index("row_index")
    df = base.loc[row_index_all].reset_index()

    xg_raw = df.loc[:, geometry_cols].to_numpy(dtype=float)
    xp_raw = df.loc[:, physics_cols].to_numpy(dtype=float)

    g_mean = np.asarray(bundle["geom_scaler_mean"], dtype=np.float32)
    g_scale = np.asarray(bundle["geom_scaler_scale"], dtype=np.float32)
    p_mean = np.asarray(bundle["phys_scaler_mean"], dtype=np.float32)
    p_scale = np.asarray(bundle["phys_scaler_scale"], dtype=np.float32)

    xg_scaled = ((xg_raw - g_mean) / g_scale).astype(np.float32)
    g_embed = _encode_geom(model, xg_scaled)

    phase1_model = joblib.load(phase1_model_path)
    phase1_meta = json.loads(phase1_meta_path.read_text(encoding="utf-8"))
    phase1_feature_cols = tuple(phase1_meta["feature_cols"])
    phase1_target_cols = tuple(phase1_meta["target_cols"])

    phase1_feature_indices = np.asarray([geometry_cols.index(c) for c in phase1_feature_cols], dtype=int)

    shared_target_cols = tuple([c for c in physics_cols if c in phase1_target_cols])
    shared_phys_indices = np.asarray([physics_cols.index(c) for c in shared_target_cols], dtype=int)
    shared_phase1_indices = np.asarray([phase1_target_cols.index(c) for c in shared_target_cols], dtype=int)

    physics_scales_map = {str(k): float(v) for k, v in dict(bundle["physics_scales"]).items()}
    physics_scales = np.asarray([physics_scales_map[c] for c in physics_cols], dtype=float)
    physics_scales = np.where(physics_scales <= 1e-12, 1.0, physics_scales)
    shared_scales = np.asarray([physics_scales_map[c] for c in shared_target_cols], dtype=float)
    shared_scales = np.where(shared_scales <= 1e-12, 1.0, shared_scales)

    physics_medians = {str(k): float(v) for k, v in dict(bundle["physics_medians"]).items()}

    return Phase3Context(
        geometry_cols=geometry_cols,
        physics_cols=physics_cols,
        physics_scales=physics_scales,
        physics_medians=physics_medians,
        row_index_all=row_index_all,
        design_id_all=design_id_all,
        xg_raw=xg_raw,
        xp_raw=xp_raw,
        g_mean=g_mean,
        g_scale=g_scale,
        p_mean=p_mean,
        p_scale=p_scale,
        g_embed=g_embed,
        train_positions=np.asarray(bundle["train_positions"], dtype=int),
        id_test_positions=np.asarray(bundle["id_test_positions"], dtype=int),
        ood_positions=np.asarray(bundle["ood_positions"], dtype=int),
        model=model,
        phase1_model=phase1_model,
        phase1_feature_cols=phase1_feature_cols,
        phase1_target_cols=phase1_target_cols,
        phase1_feature_indices=phase1_feature_indices,
        shared_target_cols=shared_target_cols,
        shared_phys_indices=shared_phys_indices,
        shared_phase1_indices=shared_phase1_indices,
        shared_scales=shared_scales,
    )


def geometry_bounds(ctx: Phase3Context, *, low_q: float = 0.001, high_q: float = 0.999) -> Tuple[np.ndarray, np.ndarray]:
    low = np.quantile(ctx.xg_raw, low_q, axis=0)
    high = np.quantile(ctx.xg_raw, high_q, axis=0)
    low = np.minimum(low, np.min(ctx.xg_raw, axis=0))
    high = np.maximum(high, np.max(ctx.xg_raw, axis=0))
    span = np.where((high - low) <= 1e-8, 1.0, high - low)
    return low.astype(float), (low + span).astype(float)


def build_target_map(
    ctx: Phase3Context,
    *,
    freq_01_GHz: float,
    anharmonicity_GHz: float,
    EJ_GHz: Optional[float] = None,
    EC_GHz: Optional[float] = None,
    charge_sensitivity_GHz: Optional[float] = None,
) -> Dict[str, float]:
    return {
        "freq_01_GHz": float(freq_01_GHz),
        "anharmonicity_GHz": float(anharmonicity_GHz),
        "EJ_GHz": float(EJ_GHz) if EJ_GHz is not None else float(ctx.physics_medians["EJ_GHz"]),
        "EC_GHz": float(EC_GHz) if EC_GHz is not None else float(ctx.physics_medians["EC_GHz"]),
        "charge_sensitivity_GHz": (
            float(charge_sensitivity_GHz)
            if charge_sensitivity_GHz is not None
            else float(ctx.physics_medians["charge_sensitivity_GHz"])
        ),
    }


def infer_active_target_cols(
    *,
    ej_given: bool,
    ec_given: bool,
    charge_given: bool,
) -> Tuple[str, ...]:
    cols = ["freq_01_GHz", "anharmonicity_GHz"]
    if ej_given:
        cols.append("EJ_GHz")
    if ec_given:
        cols.append("EC_GHz")
    if charge_given:
        cols.append("charge_sensitivity_GHz")
    return tuple(cols)


def target_vector(ctx: Phase3Context, target_map: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(target_map[c]) for c in ctx.physics_cols], dtype=float)


def target_embedding(ctx: Phase3Context, target_map: Mapping[str, float]) -> np.ndarray:
    vec = target_vector(ctx, target_map).astype(np.float32).reshape(1, -1)
    vec_s = ((vec - ctx.p_mean) / ctx.p_scale).astype(np.float32)
    return _encode_phys(ctx.model, vec_s)[0]


def retrieve_seed_positions(
    ctx: Phase3Context,
    *,
    target_embed: np.ndarray,
    library_split: str,
    top_k: int,
) -> np.ndarray:
    if library_split not in {"train", "all"}:
        raise ValueError("library_split must be one of: train, all")
    library_pos = ctx.train_positions if library_split == "train" else np.arange(len(ctx.xg_raw), dtype=int)
    sim = ctx.g_embed[library_pos] @ target_embed
    k = min(max(1, top_k), len(library_pos))
    top_local = np.argpartition(-sim, kth=k - 1)[:k]
    top_local = top_local[np.argsort(-sim[top_local])]
    return library_pos[top_local]


def _active_target_info(
    ctx: Phase3Context,
    target_map: Mapping[str, float],
    active_target_cols: Optional[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    if active_target_cols is None:
        cols = ctx.shared_target_cols
    else:
        cols = tuple(active_target_cols)

    for c in cols:
        if c not in ctx.shared_target_cols:
            raise ValueError(f"Active target column not supported by surrogate: {c}")

    phys_idx = np.asarray([ctx.shared_target_cols.index(c) for c in cols], dtype=int)
    phase_idx = np.asarray([ctx.phase1_target_cols.index(c) for c in cols], dtype=int)
    scales = np.asarray([ctx.shared_scales[ctx.shared_target_cols.index(c)] for c in cols], dtype=float)
    scales = np.where(scales <= 1e-12, 1.0, scales)
    target = np.asarray([float(target_map[c]) for c in cols], dtype=float)
    return phys_idx, phase_idx, scales, cols


def predict_phase1(ctx: Phase3Context, xg_raw: np.ndarray) -> np.ndarray:
    x = xg_raw[:, ctx.phase1_feature_indices]
    return np.asarray(ctx.phase1_model.predict(x), dtype=float)


def score_candidates(
    ctx: Phase3Context,
    *,
    candidates_raw: np.ndarray,
    target_map: Mapping[str, float],
    active_target_cols: Optional[Sequence[str]],
    target_embed: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    surrogate_weight: float,
    embedding_weight: float,
    robustness_weight: float,
    robust_samples: int,
    fabrication_tolerance: float,
    robust_std_weight: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x = np.clip(candidates_raw, low, high)

    _, active_phase_idx, active_scales, _ = _active_target_info(ctx, target_map, active_target_cols)
    active_target = np.asarray([float(target_map[c]) for c in (active_target_cols or ctx.shared_target_cols)], dtype=float)

    pred = predict_phase1(ctx, x)
    diff = (pred[:, active_phase_idx] - active_target[None, :]) / active_scales[None, :]
    surrogate_err = np.sqrt(np.mean(np.square(diff), axis=1))

    x_scaled = ((x - ctx.g_mean) / ctx.g_scale).astype(np.float32)
    z = _encode_geom(ctx.model, x_scaled, batch_size=4096)
    embedding_dist = 1.0 - (z @ target_embed)

    robust_term = np.zeros(x.shape[0], dtype=float)
    robust_mean = np.zeros(x.shape[0], dtype=float)
    robust_std = np.zeros(x.shape[0], dtype=float)
    if robust_samples > 0 and fabrication_tolerance > 0:
        jitter = rng.normal(
            loc=0.0,
            scale=max(fabrication_tolerance / 2.5, 1e-6),
            size=(x.shape[0], robust_samples, x.shape[1]),
        )
        jitter = np.clip(jitter, -fabrication_tolerance, fabrication_tolerance)
        x_pert = x[:, None, :] * (1.0 + jitter)
        x_pert = np.clip(x_pert, low[None, None, :], high[None, None, :])
        flat = x_pert.reshape(-1, x.shape[1])
        pert_pred = predict_phase1(ctx, flat)
        pert_diff = (pert_pred[:, active_phase_idx] - active_target[None, :]) / active_scales[None, :]
        pert_err = np.sqrt(np.mean(np.square(pert_diff), axis=1)).reshape(x.shape[0], robust_samples)
        robust_mean = np.mean(pert_err, axis=1)
        robust_std = np.std(pert_err, axis=1)
        robust_term = robust_mean + robust_std_weight * robust_std

    total = (
        surrogate_weight * surrogate_err
        + embedding_weight * embedding_dist
        + robustness_weight * robust_term
    )

    return {
        "total": total,
        "surrogate_err": surrogate_err,
        "embedding_dist": embedding_dist,
        "robust_mean": robust_mean,
        "robust_std": robust_std,
        "pred": pred,
        "x": x,
    }


def optimize_geometry(
    ctx: Phase3Context,
    *,
    target_map: Mapping[str, float],
    active_target_cols: Optional[Sequence[str]],
    target_embed: np.ndarray,
    seed_geometries: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    population_size: int,
    iterations: int,
    elite_fraction: float,
    explore_fraction: float,
    start_mutation_scale: float,
    end_mutation_scale: float,
    surrogate_weight: float,
    embedding_weight: float,
    robustness_weight: float,
    robust_samples: int,
    fabrication_tolerance: float,
    robust_std_weight: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    dim = seed_geometries.shape[1]
    span = np.where((high - low) <= 1e-12, 1.0, high - low)

    pop_n = max(population_size, len(seed_geometries), 16)
    pop = np.empty((pop_n, dim), dtype=float)

    seed_take = min(len(seed_geometries), pop_n)
    pop[:seed_take] = np.clip(seed_geometries[:seed_take], low, high)
    if seed_take < pop_n:
        pop[seed_take:] = rng.uniform(low=low, high=high, size=(pop_n - seed_take, dim))

    elite_n = max(2, int(round(pop_n * elite_fraction)))
    explore_n = max(2, int(round(pop_n * explore_fraction)))
    child_n = max(1, pop_n - elite_n - explore_n)

    archive_x: List[np.ndarray] = []

    for step in range(iterations):
        scores = score_candidates(
            ctx,
            candidates_raw=pop,
            target_map=target_map,
            active_target_cols=active_target_cols,
            target_embed=target_embed,
            low=low,
            high=high,
            surrogate_weight=surrogate_weight,
            embedding_weight=embedding_weight,
            robustness_weight=robustness_weight,
            robust_samples=robust_samples,
            fabrication_tolerance=fabrication_tolerance,
            robust_std_weight=robust_std_weight,
            random_state=random_state + 1009 * (step + 1),
        )
        order = np.argsort(scores["total"])
        elite = pop[order[:elite_n]]
        archive_x.append(scores["x"][order[: max(6, elite_n // 2)]])

        t = 0.0 if iterations <= 1 else (step / float(iterations - 1))
        sigma = (1.0 - t) * start_mutation_scale + t * end_mutation_scale

        parent_idx = rng.integers(0, elite_n, size=child_n)
        noise = rng.normal(loc=0.0, scale=sigma, size=(child_n, dim)) * span[None, :]
        children = np.clip(elite[parent_idx] + noise, low, high)
        explorers = rng.uniform(low=low, high=high, size=(explore_n, dim))
        pop = np.vstack([elite, children, explorers])

    merged_x = np.vstack(archive_x + [pop])
    final_scores = score_candidates(
        ctx,
        candidates_raw=merged_x,
        target_map=target_map,
        active_target_cols=active_target_cols,
        target_embed=target_embed,
        low=low,
        high=high,
        surrogate_weight=surrogate_weight,
        embedding_weight=embedding_weight,
        robustness_weight=robustness_weight,
        robust_samples=max(robust_samples, 1),
        fabrication_tolerance=fabrication_tolerance,
        robust_std_weight=robust_std_weight,
        random_state=random_state + 999_983,
    )

    dedup_key = np.round(final_scores["x"], 6)
    _, uniq_idx = np.unique(dedup_key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)

    return {
        "x": final_scores["x"][uniq_idx],
        "pred": final_scores["pred"][uniq_idx],
        "total": final_scores["total"][uniq_idx],
        "surrogate_err": final_scores["surrogate_err"][uniq_idx],
        "embedding_dist": final_scores["embedding_dist"][uniq_idx],
        "robust_mean": final_scores["robust_mean"][uniq_idx],
        "robust_std": final_scores["robust_std"][uniq_idx],
    }


def nearest_dataset_rows(ctx: Phase3Context, geometries: np.ndarray, max_count: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    scale = np.std(ctx.xg_raw, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    norm_all = ctx.xg_raw / scale[None, :]
    norm_q = geometries / scale[None, :]
    d = np.sqrt(np.sum((norm_q[:, None, :] - norm_all[None, :, :]) ** 2, axis=2))

    if max_count <= 1:
        idx = np.argmin(d, axis=1).reshape(-1, 1)
        dist = np.min(d, axis=1).reshape(-1, 1)
        return idx, dist

    k = min(max_count, d.shape[1])
    top = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    sorted_top = np.take_along_axis(top, np.argsort(np.take_along_axis(d, top, axis=1), axis=1), axis=1)
    sorted_dist = np.take_along_axis(d, sorted_top, axis=1)
    return sorted_top, sorted_dist


def run_inverse_design(
    ctx: Phase3Context,
    *,
    target_map: Mapping[str, float],
    active_target_cols: Optional[Sequence[str]] = None,
    library_split: str = "train",
    retrieval_top_k: int = 80,
    top_n: int = 10,
    population_size: int = 96,
    iterations: int = 45,
    elite_fraction: float = 0.22,
    explore_fraction: float = 0.12,
    start_mutation_scale: float = 0.15,
    end_mutation_scale: float = 0.02,
    surrogate_weight: float = 1.0,
    embedding_weight: float = 0.0,
    robustness_weight: float = 0.0,
    robust_samples: int = 0,
    fabrication_tolerance: float = 0.05,
    robust_std_weight: float = 0.30,
    bounds_low_q: float = 0.001,
    bounds_high_q: float = 0.999,
    random_state: int = 42,
) -> Dict[str, Any]:
    t_embed = target_embedding(ctx, target_map)
    seed_positions = retrieve_seed_positions(
        ctx,
        target_embed=t_embed,
        library_split=library_split,
        top_k=retrieval_top_k,
    )
    seeds = ctx.xg_raw[seed_positions]

    low, high = geometry_bounds(ctx, low_q=bounds_low_q, high_q=bounds_high_q)

    opt = optimize_geometry(
        ctx,
        target_map=target_map,
        active_target_cols=active_target_cols,
        target_embed=t_embed,
        seed_geometries=seeds,
        low=low,
        high=high,
        population_size=population_size,
        iterations=iterations,
        elite_fraction=elite_fraction,
        explore_fraction=explore_fraction,
        start_mutation_scale=start_mutation_scale,
        end_mutation_scale=end_mutation_scale,
        surrogate_weight=surrogate_weight,
        embedding_weight=embedding_weight,
        robustness_weight=robustness_weight,
        robust_samples=robust_samples,
        fabrication_tolerance=fabrication_tolerance,
        robust_std_weight=robust_std_weight,
        random_state=random_state,
    )

    order = np.argsort(opt["total"])
    take = order[: min(max(1, top_n), len(order))]

    near_idx, near_dist = nearest_dataset_rows(ctx, opt["x"][take], max_count=1)
    near_idx = near_idx[:, 0]
    near_dist = near_dist[:, 0]

    rows: List[Dict[str, Any]] = []
    for rank_idx, local_idx in enumerate(take, start=1):
        geom = opt["x"][local_idx]
        pred = opt["pred"][local_idx]
        nearest_pos = int(near_idx[rank_idx - 1])
        nearest_phys = ctx.xp_raw[nearest_pos]

        row: Dict[str, Any] = {
            "rank": int(rank_idx),
            "objective_total": float(opt["total"][local_idx]),
            "surrogate_error": float(opt["surrogate_err"][local_idx]),
            "embedding_distance": float(opt["embedding_dist"][local_idx]),
            "robust_mean_error": float(opt["robust_mean"][local_idx]),
            "robust_std_error": float(opt["robust_std"][local_idx]),
            "nearest_dataset_geom_distance": float(near_dist[rank_idx - 1]),
            "nearest_dataset_design_id": int(ctx.design_id_all[nearest_pos]),
            "nearest_dataset_row_index": int(ctx.row_index_all[nearest_pos]),
        }
        row.update({c: float(geom[i]) for i, c in enumerate(ctx.geometry_cols)})
        row.update({f"pred_{c}": float(pred[ctx.phase1_target_cols.index(c)]) for c in ctx.shared_target_cols})
        row.update({f"nearest_{c}": float(nearest_phys[ctx.physics_cols.index(c)]) for c in ctx.physics_cols})
        rows.append(row)

    active_cols = tuple(active_target_cols) if active_target_cols is not None else tuple(ctx.shared_target_cols)

    return {
        "target": {k: float(v) for k, v in dict(target_map).items()},
        "active_target_cols": list(active_cols),
        "library_split": library_split,
        "retrieval_top_k": int(retrieval_top_k),
        "optimization": {
            "population_size": int(population_size),
            "iterations": int(iterations),
            "elite_fraction": float(elite_fraction),
            "explore_fraction": float(explore_fraction),
            "start_mutation_scale": float(start_mutation_scale),
            "end_mutation_scale": float(end_mutation_scale),
            "surrogate_weight": float(surrogate_weight),
            "embedding_weight": float(embedding_weight),
            "robustness_weight": float(robustness_weight),
            "robust_samples": int(robust_samples),
            "fabrication_tolerance": float(fabrication_tolerance),
            "robust_std_weight": float(robust_std_weight),
            "bounds_low_q": float(bounds_low_q),
            "bounds_high_q": float(bounds_high_q),
        },
        "results": rows,
    }


def baseline_rerank_choice(
    ctx: Phase3Context,
    *,
    target_map: Mapping[str, float],
    active_target_cols: Optional[Sequence[str]] = None,
    library_split: str,
    retrieval_top_k: int,
) -> Dict[str, float]:
    _, active_phase_idx, active_scales, active_cols = _active_target_info(ctx, target_map, active_target_cols)
    active_target = np.asarray([float(target_map[c]) for c in active_cols], dtype=float)

    t_embed = target_embedding(ctx, target_map)
    seed_positions = retrieve_seed_positions(
        ctx,
        target_embed=t_embed,
        library_split=library_split,
        top_k=retrieval_top_k,
    )
    seeds = ctx.xg_raw[seed_positions]
    pred = predict_phase1(ctx, seeds)

    diff = (pred[:, active_phase_idx] - active_target[None, :]) / active_scales[None, :]
    score = np.sqrt(np.mean(np.square(diff), axis=1))
    best = int(np.argmin(score))
    chosen_pos = int(seed_positions[best])
    out = {"score": float(score[best]), "position": float(chosen_pos)}
    for i, name in enumerate(ctx.geometry_cols):
        out[name] = float(ctx.xg_raw[chosen_pos, i])
    for i, name in enumerate(ctx.physics_cols):
        out[f"physics_{name}"] = float(ctx.xp_raw[chosen_pos, i])
    return out

