#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_cols: Tuple[str, ...]
    target_cols: Tuple[str, ...]


SINGLE_SPEC = ModelSpec(
    name="single",
    feature_cols=("pad_width_um", "pad_height_um", "gap_um", "junction_area_um2"),
    target_cols=("freq_01_GHz", "anharmonicity_GHz", "EJ_GHz", "EC_GHz", "charge_sensitivity_GHz"),
)

COUPLED_SPEC = ModelSpec(
    name="coupled",
    feature_cols=("freq_q1_GHz", "freq_q2_GHz", "resonator_freq_GHz", "coupling_strength_g_GHz"),
    target_cols=("dressed_freq_q1_GHz", "dressed_freq_q2_GHz", "dispersive_shift_chi_GHz"),
)


def default_data_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (root / "Dataset" / "final_dataset_single.csv", root / "Dataset" / "final_dataset_coupled.csv")


def metric_report(y_true: np.ndarray, y_pred: np.ndarray, target_cols: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(target_cols):
        true_i = y_true[:, idx]
        pred_i = y_pred[:, idx]
        out[name] = {
            "mae": float(mean_absolute_error(true_i, pred_i)),
            "rmse": float(mean_squared_error(true_i, pred_i) ** 0.5),
            "r2": float(r2_score(true_i, pred_i)),
        }
    return out


def prepare_frame(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    required = list(spec.feature_cols + spec.target_cols)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{spec.name}: missing required columns: {missing}")

    clean = df[required].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError(f"{spec.name}: no usable rows after cleaning")
    return clean


def fit_model(
    df: pd.DataFrame,
    spec: ModelSpec,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> Tuple[MultiOutputRegressor, Dict[str, object], pd.DataFrame]:
    clean = prepare_frame(df, spec)

    x = clean[list(spec.feature_cols)].to_numpy(dtype=float)
    y = clean[list(spec.target_cols)].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    base = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
    )
    model = MultiOutputRegressor(base)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = metric_report(y_true=y_test, y_pred=y_pred, target_cols=spec.target_cols)

    pred_df = pd.DataFrame(x_test, columns=spec.feature_cols)
    for i, target in enumerate(spec.target_cols):
        pred_df[f"true_{target}"] = y_test[:, i]
        pred_df[f"pred_{target}"] = y_pred[:, i]

    summary: Dict[str, object] = {
        "model_name": spec.name,
        "feature_cols": list(spec.feature_cols),
        "target_cols": list(spec.target_cols),
        "rows_used": int(len(clean)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_estimators": int(n_estimators),
        "max_depth": None if max_depth is None else int(max_depth),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    return model, summary, pred_df


def train_one(
    csv_path: Path,
    spec: ModelSpec,
    output_dir: Path,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"{spec.name}: dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    model, summary, pred_df = fit_model(
        df=df,
        spec=spec,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{spec.name}_surrogate.joblib")
    (output_dir / f"{spec.name}_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_df.to_csv(output_dir / f"{spec.name}_test_predictions.csv", index=False)

    print(f"[{spec.name}] trained rows={summary['rows_used']} test_rows={summary['test_rows']}")
    for tgt, m in summary["metrics"].items():
        print(
            f"[{spec.name}] {tgt}: "
            f"MAE={m['mae']:.6g} RMSE={m['rmse']:.6g} R2={m['r2']:.6g}"
        )


def parse_args() -> argparse.Namespace:
    default_single, default_coupled = default_data_paths()
    parser = argparse.ArgumentParser(description="Train Phase-1 surrogate models from generated datasets")
    parser.add_argument("--single-csv", type=Path, default=default_single)
    parser.add_argument("--coupled-csv", type=Path, default=default_coupled)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-coupled", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test_size <= 0 or args.test_size >= 1:
        raise SystemExit("test-size must be between 0 and 1")

    if not args.skip_single:
        train_one(
            csv_path=args.single_csv,
            spec=SINGLE_SPEC,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
        )

    if not args.skip_coupled:
        train_one(
            csv_path=args.coupled_csv,
            spec=COUPLED_SPEC,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

