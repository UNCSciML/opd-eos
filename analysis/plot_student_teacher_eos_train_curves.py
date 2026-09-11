#!/usr/bin/env python3
"""Combine student and teacher EOS probabilities in one linear 1x4 figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


TEMPLATES = ("ttrl", "dapo")
TOKEN_STYLES = (
    ("endoftext", "<|endoftext|>", "#0072B2", "-"),
    ("im_end", "<|im_end|>", "#D55E00", "--"),
)
SOURCES = (
    ("student", "Student probability at last token"),
    ("teacher", "Teacher probability at last token"),
)


def _load_records(path: Path, source: str) -> pd.DataFrame:
    records = pd.read_csv(path)
    required = {
        "train_template",
        "step",
        f"{source}_prob_endoftext_last_token",
        f"{source}_prob_im_end_last_token",
    }
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if set(records["train_template"]) != set(TEMPLATES):
        raise ValueError(f"{path} must contain exactly the ttrl and dapo templates")
    return records


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def create_figure(student_records: pd.DataFrame, teacher_records: pd.DataFrame, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(9.1, 2.2), sharex=True, sharey=True)
    records_by_source = {"student": student_records, "teacher": teacher_records}

    for source_index, (source, y_label) in enumerate(SOURCES):
        records = records_by_source[source]
        for template_index, template in enumerate(TEMPLATES):
            ax = axes[2 * source_index + template_index]
            group = records.query("train_template == @template").sort_values("step")
            for token_name, token_label, color, linestyle in TOKEN_STYLES:
                ax.plot(
                    group["step"],
                    group[f"{source}_prob_{token_name}_last_token"],
                    label=token_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    zorder=3,
                )
            ax.set_title(f"{template.upper()} Template", pad=7)
            ax.set_xlabel("Training step")
            ax.set_xlim(1, int(max(student_records["step"].max(), teacher_records["step"].max())))
            ax.set_ylim(0.0, 1.0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            _style_axis(ax)
        axes[2 * source_index].set_ylabel(y_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.76, wspace=0.18)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-csv", required=True, type=Path)
    parser.add_argument("--teacher-csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    create_figure(
        _load_records(args.student_csv, "student"),
        _load_records(args.teacher_csv, "teacher"),
        args.output_prefix,
    )
    print(f"Wrote {args.output_prefix}.pdf and {args.output_prefix}.png")


if __name__ == "__main__":
    main()
