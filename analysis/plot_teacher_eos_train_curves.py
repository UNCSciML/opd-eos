#!/usr/bin/env python3
"""Plot teacher EOS probabilities at the student trajectory's last token."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator


TEMPLATES = ("ttrl", "dapo")
METRICS = {
    "teacher_prob_endoftext_last_token": "teacher_eos_prob/endoftext_151643/last_token",
    "teacher_prob_im_end_last_token": "teacher_eos_prob/im_end_151645/last_token",
}
STYLES = {
    "teacher_prob_endoftext_last_token": {
        "label": "Teacher P(<|endoftext|>)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "teacher_prob_im_end_last_token": {
        "label": "Teacher P(<|im_end|>)",
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
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
    step_pattern = re.compile(r"(?:^|\s)step:(\d+)(?=\s-\s)")
    patterns = {
        column: re.compile(rf"(?:^|\s-\s){re.escape(key)}:([-+0-9.eE]+)(?=\s-\s|$)")
        for column, key in METRICS.items()
    }
    records: dict[int, dict[str, float | int | str]] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            step_match = step_pattern.search(line)
            if not step_match:
                continue
            matches = {column: pattern.search(line) for column, pattern in patterns.items()}
            if not all(matches.values()):
                continue
            step = int(step_match.group(1))
            record: dict[str, float | int | str] = {"train_template": template, "step": step}
            record.update({column: float(match.group(1)) for column, match in matches.items() if match})
            if step in records and records[step] != record:
                raise ValueError(f"{path} contains conflicting complete records for step {step}")
            records[step] = record
    if not records:
        raise ValueError(f"No complete teacher EOS records found in {path}")
    return pd.DataFrame([records[step] for step in sorted(records)])


def load_runs(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    by_template = {template: path for template, path in runs}
    if len(runs) != len(TEMPLATES) or set(by_template) != set(TEMPLATES):
        raise ValueError("Provide exactly one --run for each of ttrl and dapo")
    return pd.concat(
        [parse_training_log(by_template[template], template) for template in TEMPLATES],
        ignore_index=True,
    )


def create_figure(records: pd.DataFrame, output_prefix: Path, yscale: str = "log") -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), sharex=True, sharey=True, facecolor="white")
    marker_interval = max(1, min(len(group) for _, group in records.groupby("train_template")) // 10)

    for ax, template in zip(axes, TEMPLATES):
        group = records.query("train_template == @template").sort_values("step")
        for metric, style in STYLES.items():
            ax.plot(
                group["step"],
                group[metric],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.55,
                markersize=3.0,
                markerfacecolor="white" if metric.endswith("im_end_last_token") else style["color"],
                markeredgewidth=0.65,
                markevery=marker_interval,
                zorder=3,
            )
        ax.set_yscale(yscale)
        ax.set_title(f"{template.upper()} train")
        ax.set_xlabel("Training step")
        ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    probability_values = records[list(METRICS)].to_numpy(dtype=float)
    if yscale == "log":
        positive_values = probability_values[probability_values > 0]
        lower = 10 ** np.floor(np.log10(positive_values.min()))
        upper = min(1.0, 10 ** np.ceil(np.log10(positive_values.max())))
        axes[0].set_ylim(lower, upper)
        axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    else:
        axes[0].set_ylim(0.0, float(probability_values.max()) * 1.08)
        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_ylabel("Teacher probability at last token")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.79, wspace=0.13)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_prefix.with_suffix(".csv"), index=False)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run, metavar="TEMPLATE,LOG_PATH")
    parser.add_argument("--yscale", choices=("log", "linear"), default="log")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    records = load_runs(args.run)
    create_figure(records, args.output_prefix, args.yscale)
    print(f"Wrote {len(records)} raw step records to {args.output_prefix}.csv")
    print(f"Wrote {args.output_prefix}.pdf and {args.output_prefix}.png")


if __name__ == "__main__":
    main()
