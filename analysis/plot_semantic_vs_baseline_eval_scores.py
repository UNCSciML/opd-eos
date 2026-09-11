#!/usr/bin/env python3
"""Plot checkpoint eval scores for the no-fix baseline against the semantic-class fix.

One row per model pair, one column per benchmark plus the benchmark average.
All curves are @16 under the TTRL template with an 8192-token budget.
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
ARM_STYLES = {
    "baseline": {"label": "No EOS fix", "color": "#E69F00", "marker": "o"},
    "semantic_class": {"label": "Semantic fix", "color": "#009E73", "marker": "s"},
}
PANEL_LABELS = "abcdefghijklmnop"


def parse_pair(value: str) -> dict:
    """Parse TITLE;BASELINE_CSV[;SEMANTIC_CSV].

    The semantic arm is optional: a stage whose semantic run has not been
    evaluated yet is still worth showing with its baseline alone.
    """
    parts = [part.strip() for part in value.split(";")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--pair must be TITLE;BASELINE_CSV[;SEMANTIC_CSV]")
    return {"title": parts[0], "baseline": Path(parts[1]),
            "semantic_class": Path(parts[2]) if len(parts) == 3 and parts[2] else None}


def load_scores(path: Path, arm: str, pair_title: str) -> pd.DataFrame:
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
    records.insert(0, "arm", arm)
    records.insert(0, "pair", pair_title)
    return records.reset_index(drop=True)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def build_figure(records: pd.DataFrame, pairs: list[dict], max_step: int,
                 shared_scale: bool) -> plt.Figure:
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
    rows, cols = len(pairs), len(COLUMNS)
    figure, axes = plt.subplots(rows, cols, figsize=(2.28 * cols, 2.30 * rows), squeeze=False)

    for r, pair in enumerate(pairs):
        row_max = 0.0
        for c, (key, name) in enumerate(COLUMNS):
            axis = axes[r][c]
            for arm, style in ARM_STYLES.items():
                group = records.query("pair == @pair['title'] and arm == @arm").sort_values("step")
                if group.empty:
                    continue
                values = 100.0 * group[key].to_numpy(dtype=np.float64)
                row_max = max(row_max, float(values.max()))
                axis.plot(group["step"], values, color=style["color"], marker=style["marker"],
                          markersize=3.0, linewidth=1.5, label=style["label"], zorder=3)
            axis.set_title(f"({PANEL_LABELS[r * cols + c]}) {name}", pad=6, y=1.0)
            _style_axis(axis)
            axis.set_xlim(0, max_step)
            axis.set_xticks(np.arange(40, max_step + 1, 40))
            if r == rows - 1:
                axis.set_xlabel("Checkpoint step")
        # A row whose scores stay under a few percent needs a decimal place, or
        # every tick rounds to the same integer label.
        decimals = 0 if row_max >= 5.0 else 1
        for c in range(cols):
            axes[r][c].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=decimals))
            if shared_scale:
                axes[r][c].set_ylim(-0.05 * max(row_max, 1.0), 1.12 * max(row_max, 1.0))
        axes[r][0].set_ylabel(f"{pair['title']}\nmean score @16")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.998),
                  ncol=2, frameon=False, handlelength=2.0, columnspacing=1.4)
    figure.subplots_adjust(left=0.088, right=0.99, bottom=0.10, top=0.86, wspace=0.30, hspace=0.42)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, type=parse_pair,
                        help="TITLE;BASELINE_CSV[;SEMANTIC_CSV]")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--per-row-scale", action="store_true",
                        help="share the y range within each row instead of per panel")
    args = parser.parse_args()

    records = pd.concat(
        [load_scores(pair[arm], arm, pair["title"])
         for pair in args.pair for arm in ARM_STYLES if pair[arm] is not None],
        ignore_index=True,
    )
    figure = build_figure(records, args.pair, args.max_step, args.per_row_scale)
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
            g = records.query("pair == @pair['title'] and arm == @arm").sort_values("step")
            if g.empty:
                continue
            last = g.iloc[-1]
            best = g["eval/avg_mean_score"].max()
            print(f"    {style['label']:14s} steps {g['step'].min()}-{g['step'].max()}  "
                  f"final avg {100 * last['eval/avg_mean_score']:.2f}%  best avg {100 * best:.2f}%  "
                  f"(AIME24 {100 * last['eval/AIME24/mean_score']:.1f}, "
                  f"AIME25 {100 * last['eval/AIME25/mean_score']:.1f}, "
                  f"AMC23 {100 * last['eval/AMC23/mean_score']:.1f})")
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
