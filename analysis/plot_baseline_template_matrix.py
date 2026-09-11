#!/usr/bin/env python3
"""Compare baseline/no-fix OPD evaluations across train and eval templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASKS = ("AIME24", "AIME25", "AMC23")
CONDITIONS = (("ttrl", "ttrl"), ("ttrl", "dapo"), ("dapo", "ttrl"), ("dapo", "dapo"))
COLORS = {"ttrl": "#3975D9", "dapo": "#E07A2D"}
LINESTYLES = {"ttrl": "-", "dapo": "--"}
MARKERS = {"ttrl": "o", "dapo": "s"}
PAPER_STYLES = {
    ("ttrl", "ttrl"): {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    ("ttrl", "dapo"): {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    ("dapo", "ttrl"): {"color": "#D55E00", "marker": "^", "linestyle": "-."},
    ("dapo", "dapo"): {"color": "#D55E00", "marker": "D", "linestyle": ":"},
}
STEP_ZERO_COLUMNS = {"eval_template", "step", "macro_accuracy", "output_length_mean", "truncation_rate"}


def parse_result(value: str) -> tuple[str, str, Path]:
    parts = value.split(",", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--result must be TRAIN,EVAL,RESULT_DIR")
    train_template, eval_template, result_dir = parts
    if train_template not in {"ttrl", "dapo"} or eval_template not in {"ttrl", "dapo"}:
        raise argparse.ArgumentTypeError("TRAIN and EVAL must each be ttrl or dapo")
    return train_template, eval_template, Path(result_dir)


def aggregate_condition(
    train_template: str,
    eval_template: str,
    result_dir: Path,
    official_dapo: bool = False,
) -> pd.DataFrame:
    source = result_dir / "grading_summary.csv"
    raw = pd.read_csv(source)
    required = {"step", "task", "mean_score", "output_length_mean", "truncation_rate", "num_rollouts"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    score_source = source
    score_raw = raw
    if official_dapo and eval_template == "dapo":
        score_source = result_dir / "grading_summary_official_dapo.csv"
        score_raw = pd.read_csv(score_source)
        score_required = {"step", "task", "mean_score"}
        score_missing = score_required - set(score_raw.columns)
        if score_missing:
            raise ValueError(f"{score_source} is missing columns: {sorted(score_missing)}")

    rows = []
    for step, group in raw.groupby("step", sort=True):
        if set(group["task"]) != set(TASKS) or len(group) != len(TASKS):
            raise ValueError(f"{source} step {step} does not contain exactly {TASKS}")
        score_group = score_raw[score_raw["step"] == step]
        if set(score_group["task"]) != set(TASKS) or len(score_group) != len(TASKS):
            raise ValueError(f"{score_source} step {step} does not contain exactly {TASKS}")
        weights = group["num_rollouts"].to_numpy(dtype=float)
        rows.append(
            {
                "train_template": train_template,
                "eval_template": eval_template,
                "step": int(step),
                "macro_accuracy": float(score_group["mean_score"].mean()),
                "output_length_mean": float(np.average(group["output_length_mean"], weights=weights)),
                "truncation_rate": float(np.average(group["truncation_rate"], weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def load_results(results: list[tuple[str, str, Path]], official_dapo: bool = False) -> pd.DataFrame:
    by_condition = {(train, evaluate): path for train, evaluate, path in results}
    if set(by_condition) != set(CONDITIONS) or len(results) != len(CONDITIONS):
        raise ValueError("Provide each of the four TTRL/DAPO train-eval combinations exactly once")
    return pd.concat(
        [
            aggregate_condition(
                train,
                evaluate,
                by_condition[(train, evaluate)],
                official_dapo=official_dapo,
            )
            for train, evaluate in CONDITIONS
        ],
        ignore_index=True,
    )


def append_step_zero(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    step_zero = pd.read_csv(path)
    missing = STEP_ZERO_COLUMNS.difference(step_zero.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(step_zero) != len(EVAL_TEMPLATES := {evaluate for _, evaluate in CONDITIONS}):
        raise ValueError(f"{path} must contain exactly one row per eval template")
    if set(step_zero["eval_template"]) != EVAL_TEMPLATES or set(step_zero["step"]) != {0}:
        raise ValueError(f"{path} must contain step 0 for TTRL and DAPO eval templates")
    if (records["step"] == 0).any():
        raise ValueError("Input results already contain step 0")

    rows = []
    for train_template, eval_template in CONDITIONS:
        source = step_zero[step_zero["eval_template"] == eval_template].iloc[0]
        rows.append(
            {
                "train_template": train_template,
                "eval_template": eval_template,
                "step": 0,
                "macro_accuracy": float(source["macro_accuracy"]),
                "output_length_mean": float(source["output_length_mean"]),
                "truncation_rate": float(source["truncation_rate"]),
            }
        )
    return pd.concat([pd.DataFrame(rows), records], ignore_index=True).sort_values(
        ["train_template", "eval_template", "step"], ignore_index=True
    )


def summarize(records: pd.DataFrame) -> dict:
    conditions = {}
    for (train, evaluate), group in records.groupby(["train_template", "eval_template"], sort=False):
        ordered = group.sort_values("step")
        final = ordered.iloc[-1]
        best = ordered.loc[ordered["macro_accuracy"].idxmax()]
        conditions[f"{train}_train__{evaluate}_eval"] = {
            "final_step": int(final["step"]),
            "final_macro_accuracy": float(final["macro_accuracy"]),
            "final_output_length_mean": float(final["output_length_mean"]),
            "final_truncation_rate": float(final["truncation_rate"]),
            "best_step": int(best["step"]),
            "best_macro_accuracy": float(best["macro_accuracy"]),
        }
    return {
        "score_aggregation": "unweighted mean of AIME24, AIME25, and AMC23 mean_score",
        "length_aggregation": "num_rollouts-weighted mean across AIME24, AIME25, and AMC23",
        "conditions": conditions,
    }


def _spread_labels(values: list[float], lower: float, upper: float, minimum_gap_fraction: float = 0.055) -> list[float]:
    gap = (upper - lower) * minimum_gap_fraction
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [value for value in values]
    for position in range(1, len(indexed)):
        previous_index, _ = indexed[position - 1]
        current_index, _ = indexed[position]
        adjusted[current_index] = max(adjusted[current_index], adjusted[previous_index] + gap)
    overflow = max(adjusted) - (upper - gap * 0.25)
    if overflow > 0:
        adjusted = [value - overflow for value in adjusted]
    underflow = (lower + gap * 0.25) - min(adjusted)
    if underflow > 0:
        adjusted = [value + underflow for value in adjusted]
    return adjusted


def create_figure(records: pd.DataFrame, summary: dict, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11.5,
            "axes.edgecolor": "#BFC7D2",
            "axes.linewidth": 0.9,
            "xtick.color": "#5F6978",
            "ytick.color": "#5F6978",
            "text.color": "#182233",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.3), facecolor="white")
    score_ax, length_ax = axes

    plotted = []
    for train, evaluate in CONDITIONS:
        group = records.query("train_template == @train and eval_template == @evaluate").sort_values("step")
        label = f"Train {train.upper()}  →  Eval {evaluate.upper()}"
        style = {
            "color": COLORS[train],
            "linestyle": LINESTYLES[evaluate],
            "marker": MARKERS[evaluate],
            "markersize": 5.2,
            "linewidth": 2.25,
            "markerfacecolor": "white" if evaluate == "dapo" else COLORS[train],
            "markeredgewidth": 1.25,
            "markeredgecolor": COLORS[train],
        }
        line = score_ax.plot(group["step"], 100 * group["macro_accuracy"], label=label, **style)[0]
        length_ax.plot(group["step"], group["output_length_mean"], label=label, **style)
        plotted.append((train, evaluate, group, line))

        best_index = group["macro_accuracy"].idxmax()
        best = group.loc[best_index]
        score_ax.scatter(
            [best["step"]],
            [100 * best["macro_accuracy"]],
            marker="*",
            s=120,
            color=COLORS[train],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    for ax in axes:
        ax.grid(axis="y", color="#E2E7ED", linewidth=0.8)
        ax.grid(axis="x", color="#F0F2F5", linewidth=0.55)
        include_step_zero = bool((records["step"] == 0).any())
        ax.set_xlim(-5 if include_step_zero else 15, 230)
        ax.set_xticks(([0] if include_step_zero else []) + list(np.arange(20, 201, 20)))
        ax.set_xlabel("Training checkpoint step")
        ax.spines[["top", "right"]].set_visible(False)

    score_ax.set_title("A. Evaluation accuracy", loc="left", pad=12)
    score_ax.set_ylabel("Macro accuracy (%)")
    score_values = 100 * records["macro_accuracy"]
    score_low = max(0.0, float(score_values.min()) - 1.1)
    score_high = float(score_values.max()) + 2.1
    score_ax.set_ylim(score_low, score_high)

    length_ax.set_title("B. Response-length inflation", loc="left", pad=12)
    length_ax.set_ylabel("Mean response length (tokens)")
    length_ax.set_ylim(900, 8700)
    length_ax.axhline(8192, color="#7F8998", linestyle=":", linewidth=1.5, zorder=1)
    length_ax.text(18, 8192 - 145, "8192-token generation budget", color="#6B7584", ha="left", va="top")

    score_final_values = [100 * float(group.iloc[-1]["macro_accuracy"]) for _, _, group, _ in plotted]
    score_label_values = _spread_labels(score_final_values, *score_ax.get_ylim())
    length_final_values = [float(group.iloc[-1]["output_length_mean"]) for _, _, group, _ in plotted]
    length_label_values = _spread_labels(length_final_values, *length_ax.get_ylim())

    for index, (train, evaluate, group, _) in enumerate(plotted):
        final = group.iloc[-1]
        short_label = f"{train.upper()}→{evaluate.upper()}"
        score_ax.annotate(
            f"{short_label}  {100 * final['macro_accuracy']:.2f}%",
            xy=(final["step"], 100 * final["macro_accuracy"]),
            xytext=(207, score_label_values[index]),
            ha="left",
            va="center",
            fontsize=9.3,
            color=COLORS[train],
            arrowprops={"arrowstyle": "-", "color": COLORS[train], "linewidth": 0.8, "alpha": 0.65},
        )
        length_ax.annotate(
            f"{short_label}  {final['output_length_mean']:,.0f}\n({100 * final['truncation_rate']:.1f}% trunc.)",
            xy=(final["step"], final["output_length_mean"]),
            xytext=(207, length_label_values[index]),
            ha="left",
            va="center",
            fontsize=9.1,
            linespacing=1.2,
            color=COLORS[train],
            arrowprops={"arrowstyle": "-", "color": COLORS[train], "linewidth": 0.8, "alpha": 0.65},
        )

    handles, labels = score_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Baseline OPD: train-template × eval-template comparison",
        x=0.055,
        y=0.99,
        ha="left",
        va="top",
        fontsize=17.5,
        weight="bold",
    )
    fig.text(
        0.055,
        0.947,
        "No EOS fix · sampled-token OPD · @16 · temperature 0.7 · max generation 8192",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#657082",
    )
    fig.text(
        0.055,
        0.018,
        "Macro accuracy is the unweighted mean of AIME24, AIME25, and AMC23; length and truncation are rollout-weighted.  ★ best checkpoint per curve",
        ha="left",
        va="bottom",
        fontsize=9.4,
        color="#657082",
    )
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.07, right=0.92, wspace=0.27)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def create_paper_figure(records: pd.DataFrame, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (score_ax, length_ax) = plt.subplots(1, 2, figsize=(6.8, 2.8), facecolor="white")

    for train, evaluate in CONDITIONS:
        group = records.query("train_template == @train and eval_template == @evaluate").sort_values("step")
        style = PAPER_STYLES[(train, evaluate)]
        label = f"{train.upper()}$\\rightarrow${evaluate.upper()}"
        plot_kwargs = {
            **style,
            "label": label,
            "linewidth": 1.7,
            "markersize": 4.2,
            "markeredgewidth": 0.7,
            "markerfacecolor": "white" if evaluate == "dapo" else style["color"],
            "markevery": max(1, len(group) // 10),
            "zorder": 3,
        }
        score_ax.plot(group["step"], 100 * group["macro_accuracy"], **plot_kwargs)
        length_ax.plot(group["step"], group["output_length_mean"], **plot_kwargs)

    for ax in (score_ax, length_ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.20)
        ax.set_axisbelow(True)
        include_step_zero = bool((records["step"] == 0).any())
        ax.set_xlim(-5 if include_step_zero else 15, 205)
        ax.set_xticks(([0] if include_step_zero else []) + [20, 80, 140, 200])
        ax.set_xlabel("Training Steps")

    score_values = 100 * records["macro_accuracy"]
    score_padding = max(0.5, 0.06 * float(score_values.max() - score_values.min()))
    score_ax.set_ylim(float(score_values.min()) - score_padding, float(score_values.max()) + score_padding)
    score_ax.set_ylabel("Avg@16 (%)")
    score_ax.set_title("(a) Accuracy", pad=4)
    score_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

    length_values = records["output_length_mean"]
    length_padding = max(200.0, 0.04 * float(length_values.max() - length_values.min()))
    length_ax.set_ylim(
        max(0.0, float(length_values.min()) - length_padding),
        float(length_values.max()) + length_padding,
    )
    length_ax.set_ylabel("Avg. Response Length")
    length_ax.set_title("(b) Length", pad=4)
    length_ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
    length_ax.axhline(8192, color="#666666", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)
    length_ax.annotate(
        "8192 budget",
        xy=(22, 8192),
        xytext=(0, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    handles, labels = score_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.3, w_pad=1.0)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_result,
        required=True,
        help="TRAIN,EVAL,RESULT_DIR; provide all four TTRL/DAPO combinations",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Render a compact ACL/EMNLP-style 6.8-inch figure at 600 DPI",
    )
    parser.add_argument(
        "--official-dapo",
        action="store_true",
        help="Use grading_summary_official_dapo.csv for the accuracy of DAPO-template evaluations",
    )
    parser.add_argument(
        "--step-zero-csv",
        type=Path,
        help="Prepend a shared base-model step 0 row for each TTRL/DAPO eval template",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_results(args.result, official_dapo=args.official_dapo)
    if args.step_zero_csv is not None:
        records = append_step_zero(records, args.step_zero_csv)
    summary = summarize(records)
    summary["dapo_scoring"] = "official" if args.official_dapo else "semantic"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.paper_style:
        create_paper_figure(records, args.output_prefix)
    else:
        create_figure(records, summary, args.output_prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
