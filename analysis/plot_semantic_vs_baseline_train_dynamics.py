#!/usr/bin/env python3
"""Plot TTRL length dynamics for the no-fix baseline against the semantic-class fix.

One column pair per model: mean response length and clip rate, with the teacher's
own rollout statistics on the same prompt stream drawn as a dashed reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


METRICS = {
    "response_length_mean": "response_length/mean",
    "response_length_clip_ratio": "response_length/clip_ratio",
}
ARM_STYLES = {
    "baseline": {"label": "No EOS fix", "color": "#E69F00"},
    "semantic_class": {"label": "Semantic fix", "color": "#009E73"},
}
TEACHER_COLOR = "#555555"
BUDGET_TOKENS = 7168.0


def parse_pair(value: str) -> dict[str, str]:
    """Parse TITLE,BASELINE_CSV,SEMANTIC_CSV,TEACHER_SUMMARY_JSON."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--pair must be TITLE,BASELINE_CSV,SEMANTIC_CSV,TEACHER_SUMMARY_JSON"
        )
    return {
        "title": parts[0],
        "baseline": Path(parts[1]),
        "semantic_class": Path(parts[2]),
        "teacher": Path(parts[3]),
    }


def load_history(path: Path, arm: str, pair_title: str) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    step_column = "training/global_step" if "training/global_step" in history else "_step"
    required = [step_column, *METRICS.values()]
    missing = [column for column in required if column not in history]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    records = history[required].rename(
        columns={step_column: "step", **{source: target for target, source in METRICS.items()}}
    )
    numeric_columns = ["step", *METRICS]
    records[numeric_columns] = records[numeric_columns].apply(pd.to_numeric, errors="coerce")
    records = records.dropna(subset=numeric_columns)
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    if records.empty:
        raise ValueError(f"{path} contains no complete training records")
    records.insert(0, "arm", arm)
    records.insert(0, "pair", pair_title)
    return records.reset_index(drop=True)


def load_teacher_reference(path: Path) -> dict[str, float]:
    summary = json.loads(Path(path).read_text())
    for key in ("response_length_mean", "length_capped_fraction"):
        if key not in summary:
            raise ValueError(f"{path} is missing {key}")
    return {
        "response_length_mean": float(summary["response_length_mean"]),
        "clip_ratio": float(summary["length_capped_fraction"]),
    }


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
    records: pd.DataFrame,
    pairs: list[dict],
    teachers: list[dict[str, float]],
    smoothing_weight: float,
    max_step: int,
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
    figure, axes = plt.subplots(1, 4, figsize=(9.1, 2.45), sharex=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")

    for pair_index, (pair, teacher) in enumerate(zip(pairs, teachers)):
        length_axis = axes[2 * pair_index]
        clip_axis = axes[2 * pair_index + 1]

        for arm, style in ARM_STYLES.items():
            group = records.query("pair == @pair['title'] and arm == @arm").sort_values("step")
            if group.empty:
                continue
            for axis, column in (
                (length_axis, "response_length_mean"),
                (clip_axis, "response_length_clip_ratio"),
            ):
                values = group[column].to_numpy(dtype=np.float64)
                if column == "response_length_clip_ratio":
                    values = 100.0 * values
                axis.plot(
                    group["step"], values, color=style["color"],
                    linewidth=0.75, alpha=0.18, zorder=1,
                )
                axis.plot(
                    group["step"], wandb_ema(values, smoothing_weight=smoothing_weight),
                    label=style["label"], color=style["color"],
                    linewidth=1.65, alpha=1.0, zorder=3,
                )

        length_axis.set_title(
            f"{panel_labels[2 * pair_index]} Mean response length\n{pair['title']}", pad=7, y=1.0
        )
        length_axis.set_ylabel("Tokens")
        length_axis.set_ylim(0, BUDGET_TOKENS * 1.055)
        length_axis.axhline(BUDGET_TOKENS, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
        length_axis.text(
            max_step - 4.0, BUDGET_TOKENS, f"{BUDGET_TOKENS:.0f}-token budget",
            fontsize=6.4, color="#666666", ha="right", va="bottom",
        )
        length_axis.axhline(
            teacher["response_length_mean"], color=TEACHER_COLOR,
            linestyle="--", linewidth=1.1, zorder=2,
        ).set_gid("teacher-reference")

        clip_axis.set_title(
            f"{panel_labels[2 * pair_index + 1]} Clipped responses\n{pair['title']}", pad=7, y=1.0
        )
        clip_axis.set_ylabel("Clip rate (%)")
        clip_axis.set_ylim(-3, 103)
        clip_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        clip_axis.axhline(
            100.0 * teacher["clip_ratio"], color=TEACHER_COLOR,
            linestyle="--", linewidth=1.1, zorder=2,
        ).set_gid("teacher-reference")

    for axis in axes:
        _style_axis(axis)
        axis.set_xlim(0, max_step)
        axis.set_xticks(np.arange(40, max_step + 1, 40))
        axis.set_xlabel("Training step")

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=TEACHER_COLOR, linestyle="--", linewidth=1.1))
    labels.append("Teacher on the same prompts")
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, handlelength=2.0, columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.207, top=0.68, wspace=0.39)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", required=True, type=parse_pair,
        help="TITLE,BASELINE_CSV,SEMANTIC_CSV,TEACHER_SUMMARY_JSON (give exactly two)",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-weight", type=float, default=0.8)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    if len(args.pair) != 2:
        parser.error("provide exactly two --pair arguments")

    frames, teachers = [], []
    for pair in args.pair:
        teachers.append(load_teacher_reference(pair["teacher"]))
        for arm in ARM_STYLES:
            frames.append(load_history(pair[arm], arm, pair["title"]))
    records = pd.concat(frames, ignore_index=True)

    figure = build_figure(records, args.pair, teachers, args.smoothing_weight, args.max_step)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{args.output_prefix}.{suffix}", dpi=600 if suffix == "png" else None,
                       bbox_inches="tight")
    plt.close(figure)

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(args.csv_output, index=False)

    for pair, teacher in zip(args.pair, teachers):
        print(f"{pair['title']}: teacher length {teacher['response_length_mean']:.1f} tokens, "
              f"clip {100.0 * teacher['clip_ratio']:.2f}%")
        for arm in ARM_STYLES:
            group = records.query("pair == @pair['title'] and arm == @arm")
            print(f"    {ARM_STYLES[arm]['label']:14s} steps {group['step'].min()}-"
                  f"{group['step'].max()}  final length "
                  f"{group.iloc[-1]['response_length_mean']:.0f}  final clip "
                  f"{100.0 * group.iloc[-1]['response_length_clip_ratio']:.1f}%")
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
