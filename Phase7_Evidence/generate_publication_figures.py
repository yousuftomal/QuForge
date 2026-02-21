import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate publication figure pack from Phase 7 outputs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "Phase7_Evidence" / "artifacts_large_sweep",
        help="Directory containing phase7_multiseed_raw.csv and related outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Phase7_Evidence" / "artifacts_large_sweep" / "figures",
        help="Directory to write figure pack",
    )
    return parser.parse_args()


def ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def make_rank_stability(summary: pd.DataFrame, outdir: Path) -> Path:
    df = summary.copy().sort_values("reliability_rank_mean", ascending=True)

    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))
    x = np.arange(len(df))

    bars = ax1.bar(
        x,
        df["reliability_rank_mean"],
        yerr=df["reliability_rank_std"],
        capsize=5,
        color="#2a9d8f",
        edgecolor="#1f6f67",
        alpha=0.9,
        label="Mean reliability rank (lower is better)",
    )
    ax1.set_ylabel("Reliability Rank")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["variant"], rotation=20, ha="right")
    ax1.set_ylim(0, max(6.0, float(df["reliability_rank_mean"].max()) + 1.2))

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["top1_count"],
        marker="o",
        linewidth=2,
        color="#e76f51",
        label="Top-1 seed count",
    )
    ax2.set_ylabel("Top-1 Count")
    ax2.set_ylim(0, max(10, int(df["top1_count"].max()) + 1))

    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_title("Figure 1. Reliability Rank Stability Across Seeds")

    lines, labels = [], []
    for ax in [ax1, ax2]:
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="upper right")

    for rect, v in zip(bars, df["reliability_rank_mean"]):
        ax1.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = outdir / "figure1_rank_stability.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ood_boxplots(raw: pd.DataFrame, outdir: Path) -> Path:
    df = raw.copy()
    plot_df = pd.concat(
        [
            df[["variant", "seed", "ood_t1_mae"]].rename(columns={"ood_t1_mae": "value"}).assign(metric="OOD T1 MAE (us)"),
            df[["variant", "seed", "ood_t2_log10_mae"]].rename(columns={"ood_t2_log10_mae": "value"}).assign(metric="OOD T2 log10 MAE"),
        ],
        ignore_index=True,
    )

    variant_order = (
        df.groupby("variant", as_index=False)["reliability_rank"]
        .mean()
        .sort_values("reliability_rank", ascending=True)["variant"]
        .tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))

    sub1 = plot_df[plot_df["metric"] == "OOD T1 MAE (us)"]
    sub2 = plot_df[plot_df["metric"] == "OOD T2 log10 MAE"]

    sns.boxplot(data=sub1, x="variant", y="value", order=variant_order, ax=axes[0], color="#8ecae6")
    sns.stripplot(data=sub1, x="variant", y="value", order=variant_order, ax=axes[0], color="#023047", alpha=0.6, size=4)
    axes[0].set_title("OOD T1 Error Distribution")
    axes[0].set_ylabel("MAE (us)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    sns.boxplot(data=sub2, x="variant", y="value", order=variant_order, ax=axes[1], color="#ffb703")
    sns.stripplot(data=sub2, x="variant", y="value", order=variant_order, ax=axes[1], color="#9b2226", alpha=0.6, size=4)
    axes[1].set_title("OOD T2 Error Distribution")
    axes[1].set_ylabel("log10 MAE")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Figure 2. OOD Error Distributions Across 10 Seeds", y=1.02, fontsize=13)
    fig.tight_layout()
    out = outdir / "figure2_ood_boxplots.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_holdout_table(holdout: pd.DataFrame, outdir: Path) -> Path:
    df = holdout.copy()
    if "holdout_t1_mae" in df.columns:
        df["holdout_t1_mae"] = df["holdout_t1_mae"].map(lambda v: "NA" if pd.isna(v) else f"{float(v):.6f}")
    if "holdout_t2_log10_mae" in df.columns:
        df["holdout_t2_log10_mae"] = df["holdout_t2_log10_mae"].map(lambda v: "NA" if pd.isna(v) else f"{float(v):.6f}")

    show_cols = [
        "source_name",
        "source_rows",
        "source_rows_with_t1",
        "source_rows_with_t2",
        "holdout_t1_mae",
        "holdout_t2_log10_mae",
    ]
    df = df[show_cols]

    fig, ax = plt.subplots(figsize=(13.8, max(2.8, 1.2 + 0.62 * len(df))))
    ax.axis("off")
    ax.set_title("Figure 3. Source Holdout Error Table", loc="left", fontsize=13, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#264653")
        else:
            cell.set_facecolor("#f4f6f8" if row % 2 == 0 else "#ffffff")

    fig.tight_layout()
    out = outdir / "figure3_source_holdout_table.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_caption_md(outdir: Path) -> Path:
    lines = [
        "# Figure Captions",
        "",
        "## Figure 1: Reliability Rank Stability Across Seeds",
        "Shows each variant's mean reliability rank with standard deviation error bars over the 10-seed sweep. The red line marks how many seeds each variant ranked first.",
        "",
        "## Figure 2: OOD Error Distributions Across 10 Seeds",
        "Boxplots and seed-level points for OOD `t1` MAE and OOD `t2` log10 MAE, comparing reliability variants under identical seed runs.",
        "",
        "## Figure 3: Source Holdout Error Table",
        "Leave-one-source-out holdout errors for mapped measurement sources, reporting holdout row counts and error metrics.",
    ]
    p = outdir / "figure_captions.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> int:
    args = parse_args()
    in_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = in_dir / "phase7_multiseed_raw.csv"
    summary_csv = in_dir / "phase7_multiseed_summary.csv"
    holdout_csv = in_dir / "source_holdout" / "phase7_source_holdout_summary.csv"

    if not raw_csv.exists() or not summary_csv.exists() or not holdout_csv.exists():
        raise SystemExit("Missing required Phase 7 input files. Run Phase 7 first.")

    raw = pd.read_csv(raw_csv)
    summary = pd.read_csv(summary_csv)
    holdout = pd.read_csv(holdout_csv)

    ensure_cols(raw, ["variant", "seed", "ood_t1_mae", "ood_t2_log10_mae", "reliability_rank"], "phase7_multiseed_raw")
    ensure_cols(summary, ["variant", "reliability_rank_mean", "reliability_rank_std", "top1_count"], "phase7_multiseed_summary")
    ensure_cols(holdout, ["source_name", "source_rows", "holdout_t1_mae", "holdout_t2_log10_mae"], "phase7_source_holdout_summary")

    sns.set_theme(style="whitegrid", context="talk")

    f1 = make_rank_stability(summary, out_dir)
    f2 = make_ood_boxplots(raw, out_dir)
    f3 = make_holdout_table(holdout, out_dir)
    cap = write_caption_md(out_dir)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(in_dir.resolve()),
        "outputs": {
            "figure1_rank_stability": str(f1.resolve()),
            "figure2_ood_boxplots": str(f2.resolve()),
            "figure3_source_holdout_table": str(f3.resolve()),
            "figure_captions": str(cap.resolve()),
        },
    }

    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Phase 7 Publication Figure Pack Complete ===")
    print(f"figure1={f1}")
    print(f"figure2={f2}")
    print(f"figure3={f3}")
    print(f"captions={cap}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
