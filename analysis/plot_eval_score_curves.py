#!/usr/bin/env python3
"""Plot checkpoint eval scores for an arbitrary set of runs.

One panel per benchmark plus the benchmark average; one line per run. All curves
are @16 under the TTRL template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

COLUMNS = [
    ("eval/AIME24/mean_score", "AIME24"),
    ("eval/AIME25/mean_score", "AIME25"),
    ("eval/AMC23/mean_score", "AMC23"),
    ("eval/avg_mean_score", "Benchmark average"),
]
# Okabe-Ito, matching the other figures in this analysis.
PALETTE = ["#E69F00", "#009E73", "#0072B2", "#CC79A7", "#D55E00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
PANEL_LABELS = "abcdefgh"


def parse_run(value: str) -> dict:
    """Parse LABEL|HISTORY_CSV. Labels may contain commas, so the separator is a pipe."""
    parts = [part.strip() for part in value.split("|")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be LABEL|HISTORY_CSV")
    return {"label": parts[0], "path": Path(parts[1])}


def load_scores(path: Path, label: str) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    if "_step" not in history:
        raise ValueError(f"{path} has no _step column")
    wanted = [key for key, _ in COLUMNS]
    missing = [key for key in wanted if key not in history]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    records = history[["_step", *wanted]].rename(columns={"_step": "step"})
    records = records.apply(pd.to_numeric, errors="coerce").dropna(subset=["step"])
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    records.insert(0, "run", label)
    return records.reset_index(drop=True)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def build_figure(records: pd.DataFrame, runs: list[dict], max_step: int,
                 shared_scale: bool) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8.5, "axes.titlesize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cols = len(COLUMNS)
    figure, axes = plt.subplots(1, cols, figsize=(2.35 * cols, 2.45), squeeze=False)
    overall_max = 0.0

    for c, (key, name) in enumerate(COLUMNS):
        axis = axes[0][c]
        for i, run in enumerate(runs):
            group = records.query("run == @run['label']").sort_values("step")
            if group.empty:
                continue
            values = 100.0 * group[key].to_numpy(dtype=np.float64)
            overall_max = max(overall_max, float(values.max()))
            axis.plot(group["step"], values, color=PALETTE[i % len(PALETTE)],
                      marker=MARKERS[i % len(MARKERS)], markersize=3.0,
                      linewidth=1.5, label=run["label"], zorder=3)
        axis.set_title(f"({PANEL_LABELS[c]}) {name}", pad=6, y=1.0)
        _style_axis(axis)
        axis.set_xlim(0, max_step)
        axis.set_xticks(np.arange(40, max_step + 1, 40))
        axis.set_xlabel("Checkpoint step")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if shared_scale:
        for c in range(cols):
            axes[0][c].set_ylim(-0.03 * overall_max, 1.10 * overall_max)
    axes[0][0].set_ylabel("Mean score @16")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=min(len(runs), 4), frameon=False, handlelength=2.0, columnspacing=1.4)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.76, wspace=0.28)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run,
                        help="LABEL|HISTORY_CSV (repeatable)")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--shared-scale", action="store_true",
                        help="use one y range across all panels")
    args = parser.parse_args()

    records = pd.concat([load_scores(r["path"], r["label"]) for r in args.run],
                        ignore_index=True)
    figure = build_figure(records, args.run, args.max_step, args.shared_scale)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{args.output_prefix}.{suffix}",
                       dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(args.csv_output, index=False)

    for run in args.run:
        g = records.query("run == @run['label']").sort_values("step")
        best = g.loc[g["eval/avg_mean_score"].idxmax()]
        print(f"{run['label']:26s} steps {g['step'].min()}-{g['step'].max()}  "
              f"first {100 * g.iloc[0]['eval/avg_mean_score']:5.2f}%  "
              f"final {100 * g.iloc[-1]['eval/avg_mean_score']:5.2f}%  "
              f"best {100 * best['eval/avg_mean_score']:5.2f}% @ step {int(best['step'])}")
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
