#!/usr/bin/env python3
"""Plot semantic-class score and response-length curves by prompt template."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


TRAIN_TEMPLATES = ("ttrl", "dapo", "eopd")
EVAL_TEMPLATES = ("ttrl", "dapo")
TRAIN_STYLES = {
    "ttrl": {"label": "TTRL train", "color": "#0072B2"},
    "dapo": {"label": "DAPO train", "color": "#D55E00"},
    "eopd": {"label": "Raw-question train", "color": "#009E73"},
}
REQUIRED_COLUMNS = {
    "train_template",
    "eval_template",
    "step",
    "macro_mean_score",
    "macro_avg_output_length",
}
OFFICIAL_DAPO_COLUMNS = {"train_template", "step", "official_macro_accuracy"}
STEP_ZERO_COLUMNS = {"eval_template", "step", "macro_accuracy", "output_length_mean", "truncation_rate"}


def load_records(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(records.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    combinations = set(
        zip(records["train_template"].astype(str), records["eval_template"].astype(str))
    )
    expected = {(train, evaluate) for train in TRAIN_TEMPLATES for evaluate in EVAL_TEMPLATES}
    if combinations != expected:
        raise ValueError("Input must contain every train-template/eval-template combination")
    return records.sort_values(["eval_template", "train_template", "step"]).reset_index(drop=True)


def apply_official_dapo_scores(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    official = pd.read_csv(path)
    missing = OFFICIAL_DAPO_COLUMNS.difference(official.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if official.duplicated(["train_template", "step"]).any():
        raise ValueError(f"{path} contains duplicate train-template/step rows")

    dapo_mask = records["eval_template"].eq("dapo")
    expected_keys = set(zip(records.loc[dapo_mask, "train_template"], records.loc[dapo_mask, "step"]))
    official_keys = set(zip(official["train_template"], official["step"]))
    if official_keys != expected_keys:
        raise ValueError("Official DAPO scores must match every DAPO-eval train-template/step row exactly")

    score_by_key = official.set_index(["train_template", "step"])["official_macro_accuracy"]
    updated = records.copy()
    updated.loc[dapo_mask, "macro_mean_score"] = [
        score_by_key.loc[(train, step)]
        for train, step in zip(updated.loc[dapo_mask, "train_template"], updated.loc[dapo_mask, "step"])
    ]
    return updated


def append_step_zero(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    step_zero = pd.read_csv(path)
    missing = STEP_ZERO_COLUMNS.difference(step_zero.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if len(step_zero) != len(EVAL_TEMPLATES):
        raise ValueError(f"{path} must contain exactly one row per eval template")
    if set(step_zero["eval_template"]) != set(EVAL_TEMPLATES) or set(step_zero["step"]) != {0}:
        raise ValueError(f"{path} must contain step 0 for TTRL and DAPO eval templates")
    if (records["step"] == 0).any():
        raise ValueError("Input records already contain step 0")

    rows = []
    for eval_template in EVAL_TEMPLATES:
        source = step_zero[step_zero["eval_template"] == eval_template].iloc[0]
        for train_template in TRAIN_TEMPLATES:
            row = {
                column: source[column]
                if column in source.index and not pd.isna(source[column])
                else (float("nan") if pd.api.types.is_numeric_dtype(records[column]) else None)
                for column in records.columns
            }
            row.update(
                {
                    "train_template": train_template,
                    "eval_template": eval_template,
                    "step": 0,
                    "macro_mean_score": float(source["macro_accuracy"]),
                    "macro_avg_output_length": float(
                        source.get("macro_avg_output_length", source["output_length_mean"])
                    ),
                }
            )
            if "macro_truncation_rate" in row:
                row["macro_truncation_rate"] = float(
                    source.get("macro_truncation_rate", source["truncation_rate"])
                )
            rows.append(row)
    return pd.concat([pd.DataFrame(rows), records], ignore_index=True).sort_values(
        ["eval_template", "train_template", "step"], ignore_index=True
    )


def _style_axis(
    axis: plt.Axes,
    include_step_zero: bool = False,
    shade_late_window: bool = True,
) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)
    axis.set_xlim(-5 if include_step_zero else 12, 208)
    axis.set_xticks(([0] if include_step_zero else []) + list(range(20, 201, 20)))
    axis.set_xlabel("Training checkpoint step")
    if shade_late_window:
        axis.axvspan(120, 200, color="#EAF3FA", alpha=0.65, zorder=0)


def build_figure(records: pd.DataFrame, shade_late_window: bool = True) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
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
    figure, axes = plt.subplots(1, 4, figsize=(13.8, 2.8), facecolor="white")
    panels = (
        ("ttrl", "macro_mean_score", "Macro score"),
        ("ttrl", "macro_avg_output_length", "Mean response length (tokens)"),
        ("dapo", "macro_mean_score", "Macro score"),
        ("dapo", "macro_avg_output_length", "Mean response length (tokens)"),
    )

    include_step_zero = bool((records["step"] == 0).any())
    for axis, (eval_template, metric, y_label) in zip(axes, panels):
        for train_template in TRAIN_TEMPLATES:
            group = records.query(
                "train_template == @train_template and eval_template == @eval_template"
            ).sort_values("step")
            style = TRAIN_STYLES[train_template]
            axis.plot(
                group["step"],
                group[metric],
                label=style["label"],
                color=style["color"],
                marker="o",
                linewidth=1.65,
                markersize=3.4,
                zorder=3,
            )
        axis.set_title(f"Fixed {eval_template.upper()} eval template", pad=7)
        axis.set_ylabel(y_label)
        _style_axis(
            axis,
            include_step_zero=include_step_zero,
            shade_late_window=shade_late_window,
        )

    score_values = records["macro_mean_score"].astype(float)
    score_padding = max(0.005, 0.08 * (score_values.max() - score_values.min()))
    score_limits = (max(0.0, score_values.min() - score_padding), score_values.max() + score_padding)
    length_upper = float(records["macro_avg_output_length"].max()) * 1.08
    for axis in (axes[0], axes[2]):
        axis.set_ylim(*score_limits)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for axis in (axes[1], axes[3]):
        axis.set_ylim(0.0, length_upper)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
    )
    figure.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.91, wspace=0.31)
    return figure


def create_figure(
    records: pd.DataFrame,
    output_prefix: Path,
    shade_late_window: bool = True,
) -> None:
    figure = build_figure(records, shade_late_window=shade_late_window)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white"
    )
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--official-dapo-csv",
        type=Path,
        help="Replace DAPO-eval macro scores from train_template,step,official_macro_accuracy rows",
    )
    parser.add_argument(
        "--step-zero-csv",
        type=Path,
        help="Prepend a shared base-model step 0 row for each TTRL/DAPO eval template",
    )
    parser.add_argument(
        "--no-late-window-shading",
        action="store_true",
        help="Do not shade the step 120-200 region",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    records = load_records(args.input_csv)
    if args.official_dapo_csv is not None:
        records = apply_official_dapo_scores(records, args.official_dapo_csv)
    if args.step_zero_csv is not None:
        records = append_step_zero(records, args.step_zero_csv)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    create_figure(
        records,
        args.output_prefix,
        shade_late_window=not args.no_late_window_shading,
    )
    print(f"Wrote {args.output_prefix}.pdf and {args.output_prefix}.png")


if __name__ == "__main__":
    main()
