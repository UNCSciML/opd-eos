#!/usr/bin/env python3
"""Show which terminal token the student and the teacher each prefer.

One panel per terminal token, stacked vertically. Every panel carries four
curves: student and teacher, under the no-fix baseline and under the
semantic-class fix. The point is that the student's mass and the teacher's mass
sit on different surface forms of the same terminal act.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_COLORS = {"baseline": "#E69F00", "semantic_class": "#009E73"}
ARM_LABELS = {"baseline": "No EOS fix", "semantic_class": "Semantic fix"}
ROLE_STYLES = {"student": {"linestyle": "-"}, "teacher": {"linestyle": (0, (4, 1.6))}}
PANEL_LABELS = ("(a)", "(b)", "(c)")
LOG_FLOOR = 1e-9


def parse_token(value: str) -> dict:
    """Parse TOKEN_ID,DISPLAY_NAME."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--token must be TOKEN_ID,DISPLAY_NAME")
    return {"id": parts[0], "name": parts[1]}


def load_history(path: Path, arm: str, tokens: list[dict]) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    step_column = "training/global_step" if "training/global_step" in history else "_step"
    columns = {
        f"{role}_{token['id']}": f"{role}_eos_prob/token_{token['id']}/last_token"
        for role in ROLE_STYLES
        for token in tokens
    }
    missing = [source for source in columns.values() if source not in history]
    if missing or step_column not in history:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing) or step_column}")

    records = history[[step_column, *columns.values()]].rename(
        columns={step_column: "step", **{source: target for target, source in columns.items()}}
    )
    records = records.apply(pd.to_numeric, errors="coerce").dropna()
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    if records.empty:
        raise ValueError(f"{path} contains no complete records")
    records.insert(0, "arm", arm)
    return records.reset_index(drop=True)


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
    records: pd.DataFrame, tokens: list[dict], smoothing_weight: float,
    max_step: int, log_scale: bool, raw_alpha: float = 0.35,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.titlesize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(len(tokens), 1, figsize=(3.5, 2.35 * len(tokens)), sharex=True)
    axes = np.atleast_1d(axes)

    for index, (axis, token) in enumerate(zip(axes, tokens)):
        for arm in ARM_COLORS:
            group = records.query("arm == @arm").sort_values("step")
            if group.empty:
                continue
            for role, role_style in ROLE_STYLES.items():
                values = group[f"{role}_{token['id']}"].to_numpy(dtype=np.float64)
                if log_scale:
                    values = np.maximum(values, LOG_FLOOR)
                smoothed = (
                    np.exp(wandb_ema(np.log(values), smoothing_weight))
                    if log_scale
                    else wandb_ema(values, smoothing_weight)
                )
                axis.plot(group["step"], values, color=ARM_COLORS[arm],
                          linewidth=0.8, alpha=raw_alpha, zorder=1, **role_style)
                axis.plot(group["step"], smoothed, color=ARM_COLORS[arm],
                          linewidth=1.55, alpha=1.0, zorder=3,
                          label=f"{role.capitalize()}, {ARM_LABELS[arm]}", **role_style)
        axis.set_title(
            f"{PANEL_LABELS[index]} $e_{index + 1}$ = {token['name']}  (id {token['id']})",
            pad=6, y=1.0,
        )
        axis.set_ylabel("Probability at last token")
        _style_axis(axis)
        if log_scale:
            axis.set_yscale("log")
            axis.set_ylim(LOG_FLOOR * 0.6, 2.0)
        else:
            axis.set_ylim(-0.03, 1.03)
        axis.set_xlim(0, max_step)
        axis.set_xticks(np.arange(40, max_step + 1, 40))

    axes[-1].set_xlabel("Training step")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.998),
                  ncol=2, frameon=False, handlelength=2.2, columnspacing=1.2)
    figure.subplots_adjust(left=0.185, right=0.985, bottom=0.085, top=0.845, hspace=0.28)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--semantic", required=True, type=Path)
    parser.add_argument("--token", action="append", required=True, type=parse_token,
                        help="TOKEN_ID,DISPLAY_NAME (give one per panel)")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-weight", type=float, default=0.8)
    parser.add_argument("--raw-alpha", type=float, default=0.35,
                        help="opacity of the unsmoothed traces behind each curve")
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    records = pd.concat(
        [load_history(args.baseline, "baseline", args.token),
         load_history(args.semantic, "semantic_class", args.token)],
        ignore_index=True,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for log_scale in (False, True):
        figure = build_figure(records, args.token, args.smoothing_weight,
                              args.max_step, log_scale, args.raw_alpha)
        suffix_stem = f"{args.output_prefix}_{'log' if log_scale else 'linear'}"
        for suffix in ("png", "pdf"):
            figure.savefig(f"{suffix_stem}.{suffix}",
                           dpi=600 if suffix == "png" else None, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {suffix_stem}.png / .pdf")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(args.csv_output, index=False)

    for arm in ARM_COLORS:
        group = records.query("arm == @arm").sort_values("step")
        print(f"\n{ARM_LABELS[arm]}  steps {group['step'].min()}-{group['step'].max()}")
        final = group.iloc[-1]
        for token in args.token:
            print(f"    {token['name']:>16s} (id {token['id']:>3s})  "
                  f"student {final['student_' + token['id']]:.3e}  "
                  f"teacher {final['teacher_' + token['id']]:.3e}")


if __name__ == "__main__":
    main()
