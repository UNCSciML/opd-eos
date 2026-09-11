#!/usr/bin/env python3
"""Plot raw baseline OPD training dynamics for TTRL and DAPO templates."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator, PercentFormatter


METRICS = {
    "response_length_mean": "response_length/mean",
    "response_length_clip_ratio": "response_length/clip_ratio",
    "student_prob_endoftext_last_token": "student_eos_prob/endoftext_151643/last_token",
    "student_prob_im_end_last_token": "student_eos_prob/im_end_151645/last_token",
}
TEMPLATES = ("ttrl", "dapo")
ROW_COLORS = {"ttrl": "#0072B2", "dapo": "#D55E00"}
EOS_STYLES = {
    "student_prob_endoftext_last_token": {
        "label": "<|endoftext|>",
        "color": "#0072B2",
        "linestyle": "-",
    },
    "student_prob_im_end_last_token": {
        "label": "<|im_end|>",
        "color": "#D55E00",
        "linestyle": "--",
    },
}


def parse_run(value: str) -> tuple[str, Path]:
    parts = value.split(",", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be TEMPLATE,LOG_PATH")
    template, path = parts
    template = template.lower()
    if template not in TEMPLATES:
        raise argparse.ArgumentTypeError("TEMPLATE must be ttrl or dapo")
    return template, Path(path)


def parse_training_log(path: Path, template: str) -> pd.DataFrame:
    """Extract one complete raw record per training step from a VERL text log."""
    patterns = {
        column: re.compile(rf"(?:^|\s-\s){re.escape(key)}:([-+0-9.eE]+)(?=\s-\s|$)")
        for column, key in METRICS.items()
    }
    step_pattern = re.compile(r"(?:^|\s)step:(\d+)(?=\s-\s)")
    records: dict[int, dict[str, float | int | str]] = {}

    with path.open(errors="replace") as handle:
        for line in handle:
            step_match = step_pattern.search(line)
            if not step_match:
                continue
            values = {column: pattern.search(line) for column, pattern in patterns.items()}
            if not all(values.values()):
                continue
            step = int(step_match.group(1))
            record: dict[str, float | int | str] = {"train_template": template, "step": step}
            record.update({column: float(match.group(1)) for column, match in values.items() if match})
            if step in records and records[step] != record:
                raise ValueError(f"{path} contains conflicting complete records for step {step}")
            records[step] = record

    if not records:
        raise ValueError(f"No complete training records found in {path}")
    return pd.DataFrame([records[step] for step in sorted(records)])


def load_runs(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    by_template = {template: path for template, path in runs}
    if len(runs) != len(TEMPLATES) or set(by_template) != set(TEMPLATES):
        raise ValueError("Provide exactly one --run for each of ttrl and dapo")
    return pd.concat(
        [parse_training_log(by_template[template], template) for template in TEMPLATES],
        ignore_index=True,
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def create_figures(records: pd.DataFrame, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.2,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    training_fig, training_axes = plt.subplots(1, 4, figsize=(9.1, 2.2), sharex=True)
    for index, template in enumerate(TEMPLATES):
        group = records.query("train_template == @template").sort_values("step")
        steps = group["step"].to_numpy()
        row_color = ROW_COLORS[template]
        length_ax = training_axes[2 * index]
        clip_ax = training_axes[2 * index + 1]

        length_ax.plot(
            steps,
            group["response_length_mean"],
            color=row_color,
            linewidth=1.65,
            zorder=3,
        )
        clip_ax.plot(
            steps,
            100.0 * group["response_length_clip_ratio"],
            color=row_color,
            linewidth=1.65,
            zorder=3,
        )
        length_ax.set_ylabel(f"{template.upper()} Template\nTokens")
        clip_ax.set_ylabel("Clip rate (%)")
        length_ax.set_title(f"({chr(ord('a') + 2 * index)}) Mean response length", pad=7)
        clip_ax.set_title(f"({chr(ord('b') + 2 * index)}) Clipped responses", pad=7)

    max_step = int(records["step"].max())
    for ax in training_axes:
        _style_axis(ax)
        ax.set_xlim(1, max_step)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_xlabel("Training step")

    max_length = float(records["response_length_mean"].max())
    length_limit = max(7168.0, max_length)
    for ax in training_axes[::2]:
        ax.set_ylim(0, length_limit * 1.055)
        ax.axhline(7168.0, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    training_axes[0].text(
        max_step * 0.98,
        7168.0,
        "7168-token budget",
        fontsize=6.6,
        color="#666666",
        ha="right",
        va="bottom",
    )
    for ax in training_axes[1::2]:
        ax.set_ylim(-3, 103)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    training_fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.84, wspace=0.42)

    eos_fig, eos_axes = plt.subplots(1, 2, figsize=(5.0, 2.4), sharex=True, sharey=True)
    for index, template in enumerate(TEMPLATES):
        group = records.query("train_template == @template").sort_values("step")
        steps = group["step"].to_numpy()
        for metric, style in EOS_STYLES.items():
            eos_axes[index].plot(
                steps,
                group[metric],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                zorder=3,
            )
        eos_axes[index].set_yscale("log")
        eos_axes[index].set_ylabel("Probability")
        eos_axes[index].set_title(
            f"({chr(ord('a') + index)}) {template.upper()} Template\nStudent probability at last token",
            pad=7,
        )
        eos_axes[index].set_xlabel("Training step")
        eos_axes[index].set_xlim(1, max_step)
        eos_axes[index].xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        _style_axis(eos_axes[index])

    positive_probs = records[
        ["student_prob_endoftext_last_token", "student_prob_im_end_last_token"]
    ].to_numpy()
    positive_probs = positive_probs[positive_probs > 0]
    lower_power = np.floor(np.log10(positive_probs.min()))
    upper_power = min(0.0, np.ceil(np.log10(positive_probs.max())))
    eos_axes[0].set_ylim(10 ** lower_power, 10 ** upper_power)
    for ax in eos_axes:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))

    handles, labels = eos_axes[0].get_legend_handles_labels()
    eos_fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.1,
    )
    eos_fig.subplots_adjust(left=0.12, right=0.98, bottom=0.21, top=0.71, wspace=0.32)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_prefix.with_suffix(".csv"), index=False)
    training_prefix = output_prefix.with_name(f"{output_prefix.name}_1x4")
    eos_prefix = output_prefix.with_name(f"{output_prefix.name}_eos_1x2")
    training_fig.savefig(training_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    training_fig.savefig(
        training_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    eos_fig.savefig(eos_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    eos_fig.savefig(eos_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(training_fig)
    plt.close(eos_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="TEMPLATE,LOG_PATH",
        help="Training template and corresponding log; provide once for TTRL and once for DAPO.",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    records = load_runs(args.run)
    create_figures(records, args.output_prefix)
    print(f"Wrote {len(records)} raw step records to {args.output_prefix}.csv")
    print(f"Wrote {args.output_prefix}_1x4.pdf and {args.output_prefix}_1x4.png")
    print(f"Wrote {args.output_prefix}_eos_1x2.pdf and {args.output_prefix}_eos_1x2.png")


if __name__ == "__main__":
    main()
