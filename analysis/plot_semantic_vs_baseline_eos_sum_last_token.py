#!/usr/bin/env python3
"""Plot the student's summed terminal-token probability at the last token.

One panel per model pair, two curves per panel: the no-fix baseline against the
semantic-class fix. The probability is summed over the pair's terminal tokens,
so the two surface forms are merged rather than shown separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC = "student_eos_prob/sum/last_token"
ARM_STYLES = {
    "baseline": {"label": "No EOS fix", "color": "#E69F00"},
    "semantic_class": {"label": "Semantic fix", "color": "#009E73"},
}
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)", "(e)")


def parse_pair(value: str) -> dict:
    """Parse TITLE,BASELINE_CSV,SEMANTIC_CSV."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--pair must be TITLE,BASELINE_CSV,SEMANTIC_CSV")
    return {"title": parts[0], "baseline": Path(parts[1]), "semantic_class": Path(parts[2])}


def load_history(path: Path, arm: str, pair_title: str) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    step_column = "training/global_step" if "training/global_step" in history else "_step"
    for column in (step_column, METRIC):
        if column not in history:
            raise ValueError(f"{path} is missing required column {column}")

    records = history[[step_column, METRIC]].rename(
        columns={step_column: "step", METRIC: "eos_sum_last_token"}
    )
    records = records.apply(pd.to_numeric, errors="coerce").dropna()
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    if records.empty:
        raise ValueError(f"{path} contains no complete records for {METRIC}")
    records.insert(0, "arm", arm)
    records.insert(0, "pair", pair_title)
    return records.reset_index(drop=True)


def wandb_ema(values: np.ndarray, smoothing_weight: float) -> np.ndarray:
    """Apply W&B-style debiased exponential moving-average smoothing."""
    if not 0 <= smoothing_weight < 1:
        raise ValueError("smoothing_weight must be in [0, 1)")
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    last = 0.0
    for index, value in enumerate(values, start=1):
        last = last * smoothing_weight + (1.0 - smoothing_weight) * value
        smoothed[index - 1] = last / (1.0 - smoothing_weight**index)
    return smoothed


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def build_figure(
    records: pd.DataFrame, pairs: list[dict], smoothing_weight: float, max_step: int
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8.5,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, len(pairs), figsize=(2.28 * len(pairs), 2.45), sharey=True)
    axes = np.atleast_1d(axes)

    for index, (axis, pair) in enumerate(zip(axes, pairs)):
        for arm, style in ARM_STYLES.items():
            group = records.query("pair == @pair['title'] and arm == @arm").sort_values("step")
            if group.empty:
                continue
            values = group["eos_sum_last_token"].to_numpy(dtype=np.float64)
            axis.plot(group["step"], values, color=style["color"],
                      linewidth=0.75, alpha=0.18, zorder=1)
            axis.plot(group["step"], wandb_ema(values, smoothing_weight=smoothing_weight),
                      label=style["label"], color=style["color"],
                      linewidth=1.65, alpha=1.0, zorder=3)
        axis.set_title(f"{PANEL_LABELS[index]} {pair['title']}", pad=7, y=1.0)
        _style_axis(axis)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlim(0, max_step)
        axis.set_xticks(np.arange(40, max_step + 1, 40))
        axis.set_xlabel("Training step")

    axes[0].set_ylabel("Summed terminal-token\nprobability at last token")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                  ncol=2, frameon=False, handlelength=2.0, columnspacing=1.4)
    figure.subplots_adjust(left=0.125, right=0.995, bottom=0.207, top=0.72, wspace=0.16)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, type=parse_pair,
                        help="TITLE,BASELINE_CSV,SEMANTIC_CSV")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-weight", type=float, default=0.8)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    frames = [
        load_history(pair[arm], arm, pair["title"])
        for pair in args.pair
        for arm in ARM_STYLES
    ]
    records = pd.concat(frames, ignore_index=True)

    figure = build_figure(records, args.pair, args.smoothing_weight, args.max_step)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{args.output_prefix}.{suffix}",
                       dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(args.csv_output, index=False)

    for pair in args.pair:
        print(pair["title"])
        for arm, style in ARM_STYLES.items():
            group = records.query("pair == @pair['title'] and arm == @arm").sort_values("step")
            print(f"    {style['label']:14s} steps {group['step'].min()}-{group['step'].max()}"
                  f"  final {group.iloc[-1]['eos_sum_last_token']:.4f}"
                  f"  max {group['eos_sum_last_token'].max():.4f}")
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
