import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate extended paper figures from Phase 5/6/7 artifacts")
    parser.add_argument(
        "--phase7-dir",
        type=Path,
        default=root / "Phase7_Evidence" / "artifacts_large_sweep",
        help="Directory containing Phase 7 large-sweep outputs",
    )
    parser.add_argument(
        "--phase6-dir",
        type=Path,
        default=root / "Phase6_Reliability" / "artifacts",
        help="Directory containing Phase 6 outputs",
    )
    parser.add_argument(
        "--phase5-dir",
        type=Path,
        default=root / "Phase5_ClosedLoop" / "artifacts",
        help="Directory containing Phase 5 outputs",
    )
    parser.add_argument(
        "--single-csv",
        type=Path,
        default=root / "Dataset" / "final_dataset_single.csv",
        help="Single-device dataset CSV used for distribution plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Phase7_Evidence" / "artifacts_large_sweep" / "figures",
        help="Directory for writing extended figures",
    )
    return parser.parse_args()


def maybe_save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fig4_risk_coverage(phase6_dir: Path, outdir: Path) -> Optional[Path]:
    csv_path = phase6_dir / "phase6_risk_coverage_curves.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    df = df[df["split"] == "eval_all"].copy()
    if df.empty:
        return None

    # Compute per-target AURC values for legend labels and ordering.
    aurc_rows = []
    for (variant, target), grp in df.groupby(["variant", "target"]):
        g = grp.sort_values("coverage")
        aurc_rows.append(
            {
                "variant": str(variant),
                "target": str(target),
                "aurc": float(np.trapz(g["risk"].to_numpy(dtype=float), x=g["coverage"].to_numpy(dtype=float))),
            }
        )
    aurc_df = pd.DataFrame(aurc_rows)
    if aurc_df.empty:
        return None

    variant_order = (
        aurc_df.groupby("variant", as_index=False)["aurc"]
        .mean()
        .sort_values("aurc", ascending=True)["variant"]
        .tolist()
    )

    palette_vals = sns.color_palette("colorblind", n_colors=len(variant_order))
    palette = {v: palette_vals[i] for i, v in enumerate(variant_order)}
    highlight_variant = "hybrid_full_conf"

    target_title = {
        "t1_us": r"$T_1$ Risk-Coverage",
        "t2_us_log10": r"$T_2(\log_{10})$ Risk-Coverage",
    }
    target_ylabel = {
        "t1_us": r"Risk (mean absolute error, $\mu$s)",
        "t2_us_log10": r"Risk (mean absolute error, $\log_{10}$ units)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharex=True)
    for ax, target in zip(axes, ["t1_us", "t2_us_log10"]):
        sub = df[df["target"] == target].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        aurc_map = {
            str(r["variant"]): float(r["aurc"])
            for _, r in aurc_df[aurc_df["target"] == target].iterrows()
        }

        for variant in variant_order:
            vsub = sub[sub["variant"] == variant].sort_values("coverage")
            if vsub.empty:
                continue
            is_highlight = variant == highlight_variant
            ax.plot(
                vsub["coverage"].to_numpy(dtype=float),
                vsub["risk"].to_numpy(dtype=float),
                marker="o",
                markersize=4.2 if is_highlight else 3.0,
                linewidth=3.0 if is_highlight else 1.8,
                alpha=1.0 if is_highlight else 0.88,
                color=palette[variant],
                label=f"{variant} (AURC={aurc_map.get(variant, float('nan')):.3f})",
                zorder=4 if is_highlight else 2,
            )

        ax.set_title(target_title.get(target, f"Risk-Coverage ({target})"))
        ax.set_xlabel("Coverage")
        ax.set_ylabel(target_ylabel.get(target, "Risk"))
        ax.set_xlim(0.1, 1.0)
        ax.set_xticks(np.linspace(0.1, 1.0, 10))
        ax.grid(alpha=0.28, linewidth=0.75, linestyle="--")
        ax.legend(title="Variant (lower AURC is better)", fontsize=8, title_fontsize=9, frameon=True, loc="best")

    fig.suptitle("Risk-Coverage Curves (Eval-All Split)", y=1.02, fontsize=14)
    sns.despine(fig=fig)

    fig.tight_layout()
    out = maybe_save(outdir / "figure4_risk_coverage_curves.png")
    fig.savefig(out, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_phase5_tradeoff(phase5_dir: Path, outdir: Path) -> Optional[Path]:
    csv_path = phase5_dir / "phase5_candidate_batch.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    df["pass_all_label"] = df["pass_all"].map({True: "pass_all", False: "filtered"}).fillna("filtered")

    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    sns.scatterplot(
        data=df,
        x="objective_total",
        y="coh_t1_p10_us",
        hue="pass_all_label",
        style="pass_all_label",
        palette={"pass_all": "#2a9d8f", "filtered": "#e76f51"},
        alpha=0.85,
        s=42,
        ax=ax,
    )
    ax.set_title("Phase 5 Candidate Tradeoff")
    ax.set_xlabel("Objective (lower is better)")
    ax.set_ylabel("Predicted T1 p10 (us)")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(title="")

    fig.tight_layout()
    out = maybe_save(outdir / "figure5_phase5_candidate_tradeoff.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def fig6_passrate_by_target(phase5_dir: Path, outdir: Path) -> Optional[Path]:
    csv_path = phase5_dir / "phase5_candidate_batch.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "target_id" not in df.columns or "pass_all" not in df.columns:
        return None

    agg = (
        df.groupby("target_id", as_index=False)["pass_all"]
        .mean()
        .rename(columns={"pass_all": "pass_rate"})
        .sort_values("pass_rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    sns.barplot(data=agg, x="target_id", y="pass_rate", color="#457b9d", ax=ax)
    ax.set_title("Phase 5 Pass-All Rate by Target")
    ax.set_xlabel("Target ID")
    ax.set_ylabel("Pass-All Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    fig.tight_layout()
    out = maybe_save(outdir / "figure6_phase5_passrate_by_target.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def fig7_abstain_gain(phase7_dir: Path, outdir: Path) -> Optional[Path]:
    csv_path = phase7_dir / "phase7_multiseed_summary.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    keep = [
        "variant",
        "eval_all_t1_us_abstain20_gain_mean",
        "eval_all_t2_us_log10_abstain20_gain_mean",
    ]
    for col in keep:
        if col not in df.columns:
            return None

    plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "variant": df["variant"],
                    "metric": "T1 abstain@20 gain",
                    "value": df["eval_all_t1_us_abstain20_gain_mean"],
                }
            ),
            pd.DataFrame(
                {
                    "variant": df["variant"],
                    "metric": "T2(log10) abstain@20 gain",
                    "value": df["eval_all_t2_us_log10_abstain20_gain_mean"],
                }
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    sns.barplot(data=plot_df, x="variant", y="value", hue="metric", ax=ax)
    ax.set_title("Abstention Gain by Variant (Mean over Seeds)")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Gain")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)

    fig.tight_layout()
    out = maybe_save(outdir / "figure7_variant_abstain_gain.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def fig8_dataset_distribution(single_csv: Path, outdir: Path) -> Optional[Path]:
    if not single_csv.exists():
        return None
    df = pd.read_csv(single_csv)
    if df.empty:
        return None
    required = {"freq_01_GHz", "EJ_EC_ratio", "gap_um"}
    if not required.issubset(df.columns):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8))
    sns.histplot(df["freq_01_GHz"], bins=35, kde=True, color="#264653", ax=axes[0])
    axes[0].set_title("f01 Distribution")
    axes[0].set_xlabel("f01 (GHz)")
    axes[0].grid(alpha=0.2, linewidth=0.7)

    sns.histplot(df["EJ_EC_ratio"], bins=35, kde=True, color="#2a9d8f", ax=axes[1])
    axes[1].set_title("EJ/EC Distribution")
    axes[1].set_xlabel("EJ/EC")
    axes[1].grid(alpha=0.2, linewidth=0.7)

    sns.histplot(df["gap_um"], bins=35, kde=True, color="#e76f51", ax=axes[2])
    axes[2].set_title("Gap Distribution")
    axes[2].set_xlabel("Gap (um)")
    axes[2].grid(alpha=0.2, linewidth=0.7)

    fig.tight_layout()
    out = maybe_save(outdir / "figure8_dataset_distributions.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    sns.set_theme(style="whitegrid", context="talk")
    args = parse_args()

    outputs: Dict[str, str] = {}
    fig_map = {
        "figure4_risk_coverage_curves": fig4_risk_coverage(args.phase6_dir, args.output_dir),
        "figure5_phase5_candidate_tradeoff": fig5_phase5_tradeoff(args.phase5_dir, args.output_dir),
        "figure6_phase5_passrate_by_target": fig6_passrate_by_target(args.phase5_dir, args.output_dir),
        "figure7_variant_abstain_gain": fig7_abstain_gain(args.phase7_dir, args.output_dir),
        "figure8_dataset_distributions": fig8_dataset_distribution(args.single_csv, args.output_dir),
    }
    for key, path in fig_map.items():
        if path is not None:
            outputs[key] = str(path.resolve())

    manifest = args.output_dir / "figure_manifest_extended.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"generated": outputs}, indent=2), encoding="utf-8")

    print("=== Extended Figure Pack Complete ===")
    print(f"manifest={manifest}")
    for key, path in outputs.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
