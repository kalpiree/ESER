#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MovieLens motivation-analysis diagnostics.")
    parser.add_argument("--csv", default="motivation_analysis_rows.csv")
    parser.add_argument("--output-dir", default="motivation_plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = df[df["complete"] == "yes"].copy()
    for column in ["hr", "ndcg", "ber", "mor", "size", "u_diff", "b_diff", "item_diff", "delta_e"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    families = list(dict.fromkeys(df["scorer_family"]))
    markers = {
        "Top-k": "o",
        "Global-Thr.": "s",
        "Group-Thr.": "P",
        "Coverage-Set": "X",
        "User-Fair": "D",
        "FAR": "^",
        "FairRec-style": "v",
        "LTT": "*",
    }
    colors = dict(zip(families, plt.cm.tab10.colors[: len(families)]))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for _, row in df.iterrows():
        ax.scatter(
            row["ndcg"],
            row["ber"],
            color=colors.get(row["scorer_family"], "gray"),
            marker=markers.get(row["method_label"], "o"),
            s=90,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
    ax.set_xlabel("NDCG (higher is better)")
    ax.set_ylabel("Bad-exposure rate / BER (lower is better)")
    ax.set_title("Ranking Quality Does Not Guarantee Safe Exposure")
    family_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=family, markerfacecolor=color, markersize=8)
        for family, color in colors.items()
    ]
    method_handles = [
        plt.Line2D([0], [0], marker=marker, color="black", label=method, linestyle="None", markersize=7)
        for method, marker in markers.items()
        if method in set(df["method_label"])
    ]
    leg1 = ax.legend(handles=family_handles, title="Scorer family", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, title="Policy", loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.tight_layout()
    fig.savefig(output_dir / "ndcg_vs_ber.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "ndcg_vs_ber.pdf", bbox_inches="tight")
    plt.close(fig)

    topk = df[df["method_label"] == "Top-k"].sort_values(["scorer_family", "scorer"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(topk["scorer"], topk["ndcg"], color=[colors.get(f, "gray") for f in topk["scorer_family"]])
    axes[0].set_title("Top-k Ranking Quality")
    axes[0].set_ylabel("NDCG")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(topk["scorer"], topk["delta_e"], color=[colors.get(f, "gray") for f in topk["scorer_family"]])
    axes[1].set_title("Top-k Exposure Disparity")
    axes[1].set_ylabel("DeltaE")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "topk_quality_vs_disparity.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "topk_quality_vs_disparity.pdf", bbox_inches="tight")
    plt.close(fig)

    policy_mean = (
        df.groupby("method_label", as_index=False)[["hr", "ndcg", "ber", "mor", "size", "u_diff", "b_diff", "item_diff", "delta_e"]]
        .mean(numeric_only=True)
        .sort_values("ber")
    )
    policy_mean.to_csv(output_dir / "policy_mean_metrics.csv", index=False)
    print(f"wrote plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
