#!/usr/bin/env python3
"""Plot TTRL training dynamics for the baseline and four EOS fixes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


MODES = ("two_stop", "teacher_map", "semantic_class", "canonical")
METRICS = {
    "response_length_mean": "response_length/mean",
    "response_length_clip_ratio": "response_length/clip_ratio",
    "student_prob_endoftext_last_token": "student_eos_prob/endoftext_151643/last_token",
    "student_prob_im_end_last_token": "student_eos_prob/im_end_151645/last_token",
}
MODE_STYLES = {
    "two_stop": {"label": "Fix 1: Two-stop", "color": "#E69F00"},
    "teacher_map": {"label": "Fix 2: Teacher-map", "color": "#0072B2"},
    "semantic_class": {"label": "Fix 3: Semantic-class", "color": "#009E73"},
    "canonical": {"label": "Fix 4: Canonical", "color": "#CC79A7"},
}


def parse_run(value: str) -> tuple[str, Path]:
    parts = value.split(",", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--run must be EOS_MODE,HISTORY_CSV")
    mode, path = parts
    mode = mode.strip().lower()
    if mode not in MODES:
        raise argparse.ArgumentTypeError(f"EOS_MODE must be one of: {', '.join(MODES)}")
    return mode, Path(path)


def load_history(path: Path, mode: str) -> pd.DataFrame:
    history = pd.read_csv(path)
    step_column = "training/global_step" if "training/global_step" in history else "_step"
    required = [step_column, *METRICS.values()]
    missing = [column for column in required if column not in history]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    records = history[required].rename(
        columns={step_column: "step", **{source: target for target, source in METRICS.items()}}
    )
    records.insert(0, "eos_mode", mode)
    numeric_columns = ["step", *METRICS]
    records[numeric_columns] = records[numeric_columns].apply(pd.to_numeric, errors="coerce")
    records = records.dropna(subset=numeric_columns)
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    if records.empty:
        raise ValueError(f"{path} contains no complete training records")
    return records.reset_index(drop=True)


def load_runs(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    by_mode = {mode: path for mode, path in runs}
    if len(runs) != len(MODES) or set(by_mode) != set(MODES):
        raise ValueError("Provide exactly one --run for each EOS mode")
    return pd.concat([load_history(by_mode[mode], mode) for mode in MODES], ignore_index=True)


def wandb_ema(values: np.ndarray, smoothing_weight: float) -> np.ndarray:
    """Apply W&B-style debiased exponential moving-average smoothing."""
    if not 0 <= smoothing_weight < 1:
        raise ValueError("smoothing_weight must be in [0, 1)")
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    last = 0.0
    for index, value in enumerate(values, start=1):
        last = last * smoothing_weight + (1.0 - smoothing_weight) * value
        debias_weight = 1.0 - smoothing_weight**index
        smoothed[index - 1] = last / debias_weight
    return smoothed


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def build_figure(
    records: pd.DataFrame,
    smoothing_weight: float,
    teacher_response_length: float | None = None,
    teacher_clip_ratio: float | None = None,
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
            "legend.title_fontsize": 8.5,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(9.1, 2.45), sharex=True)
    plot_columns = (
        "response_length_mean",
        "response_length_clip_ratio",
        "student_prob_endoftext_last_token",
        "student_prob_im_end_last_token",
    )

    for mode in MODES:
        group = records.query("eos_mode == @mode").sort_values("step")
        style = MODE_STYLES[mode]
        for axis, column in zip(axes, plot_columns):
            values = group[column].to_numpy(dtype=np.float64)
            if column == "response_length_clip_ratio":
                values = 100.0 * values
            axis.plot(
                group["step"],
                values,
                color=style["color"],
                linewidth=0.75,
                alpha=0.18,
                zorder=1,
            )
            axis.plot(
                group["step"],
                wandb_ema(values, smoothing_weight=smoothing_weight),
                label=style["label"],
                color=style["color"],
                linewidth=1.65,
                alpha=1.0,
                zorder=3,
            )

    axes[0].set_title("(a) Mean response length\n ", pad=7, y=1.0)
    axes[0].set_ylabel("Tokens")
    length_limit = max(7168.0, float(records["response_length_mean"].max()))
    axes[0].set_ylim(0, length_limit * 1.055)
    axes[0].axhline(7168.0, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
    axes[0].text(
        196.0,
        7168.0,
        "7168-token budget",
        fontsize=6.4,
        color="#666666",
        ha="right",
        va="bottom",
    )
    if teacher_response_length is not None:
        teacher_length_line = axes[0].axhline(
            teacher_response_length,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        teacher_length_line.set_gid("teacher-reference")

    axes[1].set_title("(b) Clipped responses\n ", pad=7, y=1.0)
    axes[1].set_ylabel("Clip rate (%)")
    axes[1].set_ylim(-3, 103)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if teacher_clip_ratio is not None:
        teacher_clip_percent = 100.0 * teacher_clip_ratio
        teacher_clip_line = axes[1].axhline(
            teacher_clip_percent,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        teacher_clip_line.set_gid("teacher-reference")

    axes[2].set_title(
        "(c) Student probability at last token\n$e_1$ = <|endoftext|>", pad=7, y=1.0
    )
    axes[3].set_title(
        "(d) Student probability at last token\n$e_2$ = <|im_end|>", pad=7, y=1.0
    )
    axes[2].set_ylim(0.0, 1.02)
    e2_upper = float(records["student_prob_im_end_last_token"].max()) * 1.05
    axes[3].set_ylim(0.0, e2_upper)
    axes[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[2].set_ylabel("Probability")
    axes[3].set_ylabel(r"Probability ($\times 10^{-7}$)")
    axes[3].yaxis.get_offset_text().set_visible(False)

    for axis in axes:
        _style_axis(axis)
        axis.set_xlim(0, 200)
        axis.set_xticks(np.arange(40, 201, 40))
        axis.set_xlabel("Training step")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.207, top=0.68, wspace=0.39)
    return figure


def create_figure(
    records: pd.DataFrame,
    output_prefix: Path,
    csv_output: Path,
    smoothing_weight: float,
    teacher_response_length: float | None = None,
    teacher_clip_ratio: float | None = None,
) -> None:
    figure = build_figure(
        records,
        smoothing_weight=smoothing_weight,
        teacher_response_length=teacher_response_length,
        teacher_clip_ratio=teacher_clip_ratio,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(csv_output, index=False)
    figure.savefig(
        output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white"
    )
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="EOS_MODE,HISTORY_CSV",
        help="EOS mode and W&B history CSV; provide exactly one for each mode.",
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-weight", type=float, default=0.8)
    parser.add_argument("--teacher-response-length", type=float)
    parser.add_argument(
        "--teacher-clip-ratio",
        type=float,
        help="Teacher length-cap fraction in [0, 1].",
    )
    args = parser.parse_args()
    records = load_runs(args.run)
    csv_output = args.csv_output or args.output_prefix.with_suffix(".csv")
    create_figure(
        records,
        args.output_prefix,
        csv_output,
        args.smoothing_weight,
        teacher_response_length=args.teacher_response_length,
        teacher_clip_ratio=args.teacher_clip_ratio,
    )
    print(f"Wrote {len(records)} raw TTRL training records to {csv_output}")


if __name__ == "__main__":
    main()
