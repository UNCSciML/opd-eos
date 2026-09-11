#!/usr/bin/env python3
"""Compare semantic EOS fixing with no-fix OPD across train/eval templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = (("ttrl", "ttrl"), ("ttrl", "dapo"), ("dapo", "ttrl"), ("dapo", "dapo"))
PAPER_STYLES = {
    ("ttrl", "ttrl"): {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    ("ttrl", "dapo"): {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    ("dapo", "ttrl"): {"color": "#D55E00", "marker": "^", "linestyle": "-."},
    ("dapo", "dapo"): {"color": "#D55E00", "marker": "D", "linestyle": ":"},
}
METHODS = ("semantic_fix", "no_eos_fix")
METHOD_LABELS = {"semantic_fix": "Semantic fix", "no_eos_fix": "No EOS fix"}
TASK_LENGTH_COLUMNS = (
    "AIME24_avg_output_length",
    "AIME25_avg_output_length",
    "AMC23_avg_output_length",
)
TASK_ROLLOUT_WEIGHTS = np.asarray([3.0, 3.0, 4.0])


def _validate_conditions(records: pd.DataFrame, source: Path) -> None:
    keys = ["train_template", "eval_template", "step"]
    if records.duplicated(keys).any():
        raise ValueError(f"{source} contains duplicate train/eval/step rows")
    combinations = set(zip(records["train_template"], records["eval_template"]))
    if combinations != set(CONDITIONS):
        raise ValueError(f"{source} must contain all four TTRL/DAPO train/eval conditions")
    step_sets = [
        set(group["step"].astype(int))
        for _, group in records.groupby(["train_template", "eval_template"])
    ]
    if not step_sets or any(steps != step_sets[0] for steps in step_sets[1:]):
        raise ValueError(f"{source} conditions must contain identical checkpoint steps")
    if 0 not in step_sets[0]:
        raise ValueError(f"{source} must include step 0")


def load_comparison(semantic_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    semantic_raw = pd.read_csv(semantic_csv)
    semantic_required = {
        "train_template",
        "eval_template",
        "step",
        "macro_mean_score",
        *TASK_LENGTH_COLUMNS,
    }
    missing = semantic_required.difference(semantic_raw.columns)
    if missing:
        raise ValueError(f"{semantic_csv} is missing columns: {sorted(missing)}")
    semantic = semantic_raw[
        semantic_raw["train_template"].isin({"ttrl", "dapo"})
        & semantic_raw["eval_template"].isin({"ttrl", "dapo"})
    ].copy()
    semantic["method"] = "semantic_fix"
    semantic["macro_accuracy"] = semantic["macro_mean_score"].astype(float)
    semantic["output_length_mean"] = np.average(
        semantic.loc[:, TASK_LENGTH_COLUMNS].to_numpy(dtype=float),
        axis=1,
        weights=TASK_ROLLOUT_WEIGHTS,
    )
    semantic = semantic[
        ["method", "train_template", "eval_template", "step", "macro_accuracy", "output_length_mean"]
    ]
    _validate_conditions(semantic, semantic_csv)

    baseline_raw = pd.read_csv(baseline_csv)
    baseline_required = {
        "train_template",
        "eval_template",
        "step",
        "macro_accuracy",
        "output_length_mean",
    }
    missing = baseline_required.difference(baseline_raw.columns)
    if missing:
        raise ValueError(f"{baseline_csv} is missing columns: {sorted(missing)}")
    baseline = baseline_raw[
        baseline_raw["train_template"].isin({"ttrl", "dapo"})
        & baseline_raw["eval_template"].isin({"ttrl", "dapo"})
    ].copy()
    baseline["method"] = "no_eos_fix"
    baseline = baseline[
        ["method", "train_template", "eval_template", "step", "macro_accuracy", "output_length_mean"]
    ]
    _validate_conditions(baseline, baseline_csv)

    if set(semantic["step"].astype(int)) != set(baseline["step"].astype(int)):
        raise ValueError("Semantic-fix and no-fix inputs must contain identical checkpoint steps")
    return pd.concat([semantic, baseline], ignore_index=True).sort_values(
        ["method", "train_template", "eval_template", "step"], ignore_index=True
    )


def build_figure(records: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(13.6, 2.9), facecolor="white")

    for method_index, method in enumerate(METHODS):
        score_axis = axes[2 * method_index]
        length_axis = axes[2 * method_index + 1]
        for train_template, eval_template in CONDITIONS:
            group = records.query(
                "method == @method and train_template == @train_template "
                "and eval_template == @eval_template"
            ).sort_values("step")
            style = PAPER_STYLES[(train_template, eval_template)]
            label = f"{train_template.upper()}$\\rightarrow${eval_template.upper()}"
            plot_kwargs = {
                **style,
                "label": label,
                "linewidth": 1.65,
                "markersize": 4.0,
                "markeredgewidth": 0.7,
                "markerfacecolor": "white" if eval_template == "dapo" else style["color"],
                "zorder": 3,
            }
            score_axis.plot(group["step"], 100.0 * group["macro_accuracy"], **plot_kwargs)
            length_axis.plot(group["step"], group["output_length_mean"], **plot_kwargs)

        method_label = METHOD_LABELS[method]
        score_axis.set_title(f"{method_label}: Accuracy", pad=5)
        length_axis.set_title(f"{method_label}: Length", pad=5)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.20)
        axis.set_axisbelow(True)
        axis.set_xlim(-5, 205)
        axis.set_xticks([0, 20, 80, 140, 200])
        axis.set_xlabel("Training Steps")

    score_values = 100.0 * records["macro_accuracy"].astype(float)
    score_padding = max(0.5, 0.06 * float(score_values.max() - score_values.min()))
    score_limits = (float(score_values.min()) - score_padding, float(score_values.max()) + score_padding)
    for axis in (axes[0], axes[2]):
        axis.set_ylim(*score_limits)
        axis.set_ylabel("Avg@16 (%)")
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

    length_values = records["output_length_mean"].astype(float)
    length_padding = max(200.0, 0.04 * float(length_values.max() - length_values.min()))
    length_limits = (
        max(0.0, float(length_values.min()) - length_padding),
        max(8192.0 + 200.0, float(length_values.max()) + length_padding),
    )
    for axis in (axes[1], axes[3]):
        axis.set_ylim(*length_limits)
        axis.set_ylabel("Avg. Response Length")
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
        axis.axhline(8192, color="#666666", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
        axis.annotate(
            "8192 budget",
            xy=(22, 8192),
            xytext=(0, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.3,
            color="#555555",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87), pad=0.4, w_pad=1.0)
    return figure


def write_figure(records: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_prefix.with_suffix(".csv"), index=False)
    figure = build_figure(records)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-csv", required=True, type=Path)
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_comparison(args.semantic_csv, args.baseline_csv)
    write_figure(records, args.output_prefix)
    print(f"Wrote {args.output_prefix}.pdf and {args.output_prefix}.png")


if __name__ == "__main__":
    main()
