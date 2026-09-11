#!/usr/bin/env python3
"""Compare summed EOS probabilities for Qwen3 and Gemma-3 baseline OPD runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MaxNLocator


METRICS = {
    "student_sequence_mean": "student_eos_prob/sum/sequence_mean",
    "teacher_sequence_mean": "teacher_eos_prob/sum/sequence_mean",
    "student_last_token": "student_eos_prob/sum/last_token",
    "teacher_last_token": "teacher_eos_prob/sum/last_token",
    "response_length_mean": "response_length/mean",
    "max_tokens_fraction": "finish_reason/max_tokens_fraction",
}

MODEL_LABELS = {
    "qwen3": "Qwen3 1.7B Base → 4B",
    "gemma3": "Gemma-3 4B PT → 4B IT",
}

TEACHER_MEAN_COLOR = "#CC79A7"


def parse_training_log(path: Path, model: str) -> list[dict[str, float | int | str]]:
    step_pattern = re.compile(r"(?:^|\s)step:(\d+)(?=\s-\s)")
    patterns = {
        name: re.compile(rf"(?:^|\s-\s){re.escape(metric)}:([-+0-9.eE]+)(?=\s-\s|$)")
        for name, metric in METRICS.items()
    }
    records: dict[int, dict[str, float | int | str]] = {}

    with path.open(errors="replace") as handle:
        for line in handle:
            step_match = step_pattern.search(line)
            if not step_match:
                continue
            matches = {name: pattern.search(line) for name, pattern in patterns.items()}
            if not all(matches.values()):
                continue
            step = int(step_match.group(1))
            record: dict[str, float | int | str] = {"model": model, "step": step}
            record.update(
                {name: float(match.group(1)) for name, match in matches.items() if match}
            )
            if step in records and records[step] != record:
                raise ValueError(f"{path} contains conflicting records for step {step}")
            records[step] = record

    expected_steps = set(range(1, 201))
    if set(records) != expected_steps:
        missing = sorted(expected_steps.difference(records))
        raise ValueError(f"{path} does not contain exactly steps 1..200; missing={missing[:10]}")
    return [records[step] for step in sorted(records)]


def write_csv(records: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "step", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_teacher_response_length_mean(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    mean = float(summary["response_length_mean"])
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"Invalid response_length_mean in {path}: {mean}")
    return mean


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def plot_length_panel(
    ax: plt.Axes,
    steps: np.ndarray,
    lengths: np.ndarray,
    teacher_mean: float,
) -> None:
    ax.plot(steps, lengths, color="#009E73", linewidth=1.5, zorder=3)
    ax.axhline(7168.0, color="#666666", linestyle=":", linewidth=0.9)
    ax.axhline(
        teacher_mean,
        color=TEACHER_MEAN_COLOR,
        linestyle="--",
        linewidth=1.25,
        label="Teacher generation mean",
        zorder=2,
    )
    ax.text(
        float(steps.max()),
        teacher_mean + 70,
        f"Teacher mean: {teacher_mean:,.0f}",
        color=TEACHER_MEAN_COLOR,
        ha="right",
        va="bottom",
        fontsize=7,
    )


def create_figure(
    records: list[dict[str, float | int | str]],
    output_prefix: Path,
    probability_scale: str,
    teacher_length_means: dict[str, float],
) -> None:
    if probability_scale not in {"log", "linear"}:
        raise ValueError(f"Unsupported probability scale: {probability_scale}")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    by_model = {
        model: [record for record in records if record["model"] == model]
        for model in MODEL_LABELS
    }
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 7.0), sharex="col")
    series = (
        ("student", "#0072B2", "-", "Student"),
        ("teacher", "#D55E00", "--", "Teacher"),
    )

    probability_rows = (
        (0, "sequence_mean", "Sum EOS probability\n(sequence mean)"),
        (1, "last_token", "Sum EOS probability\n(last token)"),
    )
    all_probabilities: dict[int, list[float]] = {0: [], 1: []}
    for model_index, (model, title) in enumerate(MODEL_LABELS.items()):
        group = by_model[model]
        steps = np.asarray([record["step"] for record in group], dtype=float)
        for row_index, suffix, ylabel in probability_rows:
            ax = axes[row_index, model_index]
            for source, color, linestyle, label in series:
                values = np.asarray(
                    [record[f"{source}_{suffix}"] for record in group], dtype=float
                )
                selected_values = values[values > 0] if probability_scale == "log" else values
                all_probabilities[row_index].extend(selected_values.tolist())
                ax.plot(
                    steps,
                    values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.45,
                    label=label,
                    zorder=3,
                )
            ax.set_yscale(probability_scale)
            ax.set_ylabel(ylabel)
            _style_axis(ax)
        axes[0, model_index].set_title(title, pad=7)

        length_ax = axes[2, model_index]
        lengths = np.asarray([record["response_length_mean"] for record in group], dtype=float)
        plot_length_panel(length_ax, steps, lengths, teacher_length_means[model])
        length_ax.set_ylim(0, 7520)
        length_ax.set_ylabel("Mean response length\n(tokens)")
        length_ax.set_xlabel("Training step")
        _style_axis(length_ax)

    for row_index in (0, 1):
        probabilities = np.asarray(all_probabilities[row_index])
        if probability_scale == "log":
            lower = 10 ** np.floor(np.log10(probabilities.min()))
            upper = min(1.0, 10 ** np.ceil(np.log10(probabilities.max())))
        else:
            lower = 0.0
            upper = min(1.0, probabilities.max() * 1.08)
        for ax in axes[row_index]:
            ax.set_ylim(lower, upper)
            if probability_scale == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    for ax in axes.flat:
        ax.set_xlim(1, 200)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.8,
    )
    fig.suptitle(
        f"Baseline OPD: summed EOS probability and length dynamics ({probability_scale} scale)",
        y=1.025,
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.08, top=0.91, hspace=0.22, wspace=0.26)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-log", required=True, type=Path)
    parser.add_argument("--gemma-log", required=True, type=Path)
    parser.add_argument("--qwen-teacher-summary", required=True, type=Path)
    parser.add_argument("--gemma-teacher-summary", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    records = [
        *parse_training_log(args.qwen_log, "qwen3"),
        *parse_training_log(args.gemma_log, "gemma3"),
    ]
    teacher_length_means = {
        "qwen3": load_teacher_response_length_mean(args.qwen_teacher_summary),
        "gemma3": load_teacher_response_length_mean(args.gemma_teacher_summary),
    }
    write_csv(records, args.output_prefix.with_suffix(".csv"))
    for probability_scale in ("log", "linear"):
        figure_prefix = args.output_prefix.with_name(
            f"{args.output_prefix.name}_{probability_scale}"
        )
        create_figure(records, figure_prefix, probability_scale, teacher_length_means)
    print(f"Wrote {len(records)} records and log/linear figures to {args.output_prefix.parent}")


if __name__ == "__main__":
    main()
