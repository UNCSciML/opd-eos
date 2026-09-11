#!/usr/bin/env python3
"""K2: eval score against the length the checkpoint actually produces.

Left panel: benchmark average and AMC23 output length over checkpoints, on twin
axes. Right panel: the same points as a scatter, score against length, which
shows whether shorter checkpoints score better within a run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

SCORE = "eval/avg_mean_score"
LENGTH = "eval/AMC23/output_length_mean"
SCORE_COLOR = "#CC79A7"
LENGTH_COLOR = "#0072B2"
PALETTE = ["#CC79A7", "#0072B2", "#E69F00", "#009E73"]
MARKERS = ["o", "^", "s", "D"]


def parse_run(value: str) -> dict:
    """Parse LABEL|HISTORY_CSV."""
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be LABEL|HISTORY_CSV")
    return {"label": parts[0], "path": Path(parts[1])}


def load(path: Path, label: str) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    for column in ("_step", SCORE, LENGTH):
        if column not in history:
            raise ValueError(f"{path} is missing {column}")
    frame = history[["_step", SCORE, LENGTH]].rename(columns={"_step": "step"})
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.sort_values("step").drop_duplicates("step", keep="last")
    frame["step"] = frame["step"].astype(int)
    frame.insert(0, "run", label)
    return frame.reset_index(drop=True)


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)


def build_figure(frames: list[pd.DataFrame], runs: list[dict], max_step: int) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    # (a) the primary run on twin axes
    primary = frames[0]
    left = axes[0]
    left.plot(primary["step"], 100.0 * primary[SCORE], color=SCORE_COLOR, marker="o",
              markersize=3.2, linewidth=1.6, zorder=3)
    left.set_ylabel("Benchmark average @16", color=SCORE_COLOR)
    left.tick_params(axis="y", colors=SCORE_COLOR)
    left.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _style(left)
    left.spines["right"].set_linewidth(0.8)
    right = left.twinx()
    right.plot(primary["step"], primary[LENGTH], color=LENGTH_COLOR, marker="^",
               markersize=3.2, linewidth=1.6, linestyle=(0, (4, 1.6)), zorder=2)
    right.set_ylabel("Output length (tokens)", color=LENGTH_COLOR, labelpad=2)
    right.tick_params(axis="y", colors=LENGTH_COLOR)
    right.spines["top"].set_visible(False)
    left.set_xlim(0, max_step)
    left.set_xticks(np.arange(40, max_step + 1, 40))
    left.set_xlabel("Checkpoint step")
    left.set_title(f"(a) {runs[0]['label']}: score and length", pad=6, y=1.0)
    left.legend(handles=[Line2D([], [], color=SCORE_COLOR, marker="o", markersize=3.2,
                                linewidth=1.6, label="Benchmark average"),
                         Line2D([], [], color=LENGTH_COLOR, marker="^", markersize=3.2,
                                linewidth=1.6, linestyle=(0, (4, 1.6)), label="Output length")],
                loc="lower left", frameon=False, handlelength=2.0)

    # (b) score against length, every run
    scatter = axes[1]
    for i, (frame, run) in enumerate(zip(frames, runs)):
        scatter.plot(frame[LENGTH], 100.0 * frame[SCORE], color=PALETTE[i % len(PALETTE)],
                     marker=MARKERS[i % len(MARKERS)], markersize=3.6, linewidth=0.9,
                     alpha=0.85, label=run["label"], zorder=3)
        if len(frame) > 2:
            r = frame[LENGTH].corr(frame[SCORE], method="spearman")
            print(f"{run['label']:26s} rho(length, score) = {r:+.2f}")
    scatter.set_xlabel("AMC23 output length (tokens)")
    scatter.set_ylabel("Benchmark average @16", labelpad=2)
    scatter.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    scatter.set_title("(b) Score against the length produced", pad=6, y=1.0)
    _style(scatter)
    scatter.spines["right"].set_visible(False)
    scatter.legend(loc="best", frameon=False, handlelength=1.8, fontsize=7.5)

    figure.subplots_adjust(left=0.082, right=0.925, bottom=0.16, top=0.88, wspace=0.58)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run,
                        help="LABEL|HISTORY_CSV; the first is the twin-axis panel")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    frames = [load(r["path"], r["label"]) for r in args.run]
    figure = build_figure(frames, args.run, args.max_step)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{args.output_prefix}.{suffix}",
                       dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(args.csv_output, index=False)
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
