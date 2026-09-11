#!/usr/bin/env python3
"""K2-Horizon: length dynamics against terminal-token alignment, by student stage.

Same model family and the same teacher throughout; only the student's training
stage differs. The top row is the length trajectory, the bottom row is where each
side puts its terminal mass at the position that ends the response.

K2 has two terminal tokens with different jobs:
    id 1       <|ifm|endoftext|>   ends a DOCUMENT   (pretraining style)
    id 250019  <|ifm|im_end|>      ends a TURN       (chat style)
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

TOKENS = [
    ("token_1", r"$\langle$endoftext$\rangle$ (doc)", "#E69F00"),
    ("token_250019", r"$\langle$im_end$\rangle$ (turn)", "#0072B2"),
]
LENGTH_COLOR = "#009E73"
CLIP_COLOR = "#999999"
BUDGET = 7168.0
PANELS = "abcdefghijkl"


def parse_stage(value: str) -> dict:
    """Parse TITLE;HISTORY_CSV. Semicolon, so titles may contain commas."""
    parts = [part.strip() for part in value.split(";")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--stage must be TITLE;HISTORY_CSV")
    return {"title": parts[0], "path": Path(parts[1])}


def load(path: Path) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    step = "training/global_step" if "training/global_step" in history else "_step"
    keep = [step, "response_length/mean", "response_length/clip_ratio"]
    for token, _, _ in TOKENS:
        keep += [f"student_eos_prob/{token}/last_token", f"teacher_eos_prob/{token}/last_token"]
    missing = [c for c in keep if c not in history]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    frame = history[keep].rename(columns={step: "step"}).apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["step"]).sort_values("step").drop_duplicates("step", keep="last")
    frame["step"] = frame["step"].astype(int)
    return frame.reset_index(drop=True)


def panel_limit(own_max: int, cap: int) -> int:
    """Per-column x range: each stage gets its own axis, rounded up to a round
    number, so a 200-step run is not drawn on a 400-step axis with half of it empty."""
    return min(cap, max(200, -(-own_max // 100) * 100))


def xticks_for(max_step: int) -> np.ndarray:
    """Pick the smallest nice interval that keeps the axis to at most six ticks."""
    for interval in (20, 40, 50, 100, 200):
        ticks = np.arange(interval, max_step + 1, interval)
        if len(ticks) <= 6:
            return ticks
    return np.linspace(0, max_step, 5, dtype=int)[1:]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=2).mean().to_numpy()


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)


def load_teacher(path: Path) -> dict[str, float]:
    summary = json.loads(Path(path).read_text())
    return {"length": float(summary["response_length_mean"]),
            "clip": float(summary["length_capped_fraction"])}


def build_figure(stages: list[dict], frames: list[pd.DataFrame], window: int,
                 max_step: int, teacher: dict[str, float] | None = None) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    n = len(stages)
    figure, axes = plt.subplots(2, n, figsize=(2.55 * n, 4.7), squeeze=False)

    for col, (stage, frame) in enumerate(zip(stages, frames)):
        top, bottom = axes[0][col], axes[1][col]
        steps = frame["step"].to_numpy()

        top.plot(steps, smooth(frame["response_length/mean"].to_numpy(), window),
                 color=LENGTH_COLOR, linewidth=1.65, zorder=3)
        top.axhline(BUDGET, color="#777777", linestyle=":", linewidth=0.9, zorder=1)
        top.set_ylim(0, BUDGET * 1.06)
        top.set_title(f"({PANELS[col]}) {stage['title']}", pad=6, y=1.0)
        _style(top)
        top.spines["right"].set_linewidth(0.8)
        clip = top.twinx()
        clip.plot(steps, 100.0 * smooth(frame["response_length/clip_ratio"].to_numpy(), window),
                  color=CLIP_COLOR, linewidth=1.1, linestyle=(0, (3, 1.5)), zorder=2)
        clip.set_ylim(-3, 103)
        clip.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        if teacher is not None:
            # The teacher's own rollout statistics on the same prompt stream: the value
            # OPD should converge to, not zero. K2's teacher clips 41% of the time.
            top.axhline(teacher["length"], color=LENGTH_COLOR, linestyle=(0, (5, 2)),
                        linewidth=1.2, alpha=0.9, zorder=2).set_gid("teacher-length")
            clip.axhline(100.0 * teacher["clip"], color=CLIP_COLOR, linestyle=(0, (1, 1.4)),
                         linewidth=1.2, alpha=0.9, zorder=1).set_gid("teacher-clip")
        clip.spines["top"].set_visible(False)
        clip.tick_params(axis="y", labelsize=7, colors=CLIP_COLOR)
        if col == n - 1:
            clip.set_ylabel("Clip rate", color=CLIP_COLOR, fontsize=8)
        else:
            clip.set_yticklabels([])

        for token, _, color in TOKENS:
            bottom.plot(steps, smooth(frame[f"student_eos_prob/{token}/last_token"].to_numpy(), window),
                        color=color, linewidth=1.6, zorder=3)
            bottom.plot(steps, smooth(frame[f"teacher_eos_prob/{token}/last_token"].to_numpy(), window),
                        color=color, linewidth=1.3, linestyle=(0, (4, 1.6)), zorder=2)
        bottom.set_ylim(-0.06, 0.85)
        bottom.set_title(f"({PANELS[n + col]}) {stage['title']}", pad=6, y=1.0)
        _style(bottom)
        bottom.spines["right"].set_visible(False)

        limit = panel_limit(int(steps.max()), max_step)
        for axis in (top, bottom):
            axis.set_xlim(0, limit)
            axis.set_xticks(xticks_for(limit))
        bottom.set_xlabel("Training step")

    axes[0][0].set_ylabel("Mean response length\n(tokens)")
    axes[1][0].set_ylabel("EOS probability at last token")

    handles = [Line2D([], [], color=LENGTH_COLOR, linewidth=1.65, label="Mean response length"),
               Line2D([], [], color=CLIP_COLOR, linewidth=1.1, linestyle=(0, (3, 1.5)),
                      label="Clip rate")]
    handles += [Line2D([], [], color=c, linewidth=1.6, label=f"Student {n_}") for _, n_, c in TOKENS]
    handles += [Line2D([], [], color=c, linewidth=1.3, linestyle=(0, (4, 1.6)),
                       label=f"Teacher {n_}") for _, n_, c in TOKENS]
    if teacher is not None:
        handles += [Line2D([], [], color=LENGTH_COLOR, linewidth=1.2, linestyle=(0, (5, 2)),
                           label="Teacher mean length"),
                    Line2D([], [], color=CLIP_COLOR, linewidth=1.2, linestyle=(0, (1, 1.4)),
                           label="Teacher clip rate")]
    figure.legend(handles, [h.get_label() for h in handles], loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=3 if teacher is None else 4, frameon=False,
                  handlelength=2.0, columnspacing=1.4)
    figure.subplots_adjust(left=0.095, right=0.925, bottom=0.085, top=0.80, wspace=0.30, hspace=0.34)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, type=parse_stage)
    parser.add_argument("--teacher-summary", type=Path,
                        help="teacher_rollout_lengths/<run>/summary.json for the dashed reference")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    frames = [load(stage["path"]) for stage in args.stage]
    teacher = load_teacher(args.teacher_summary) if args.teacher_summary else None
    figure = build_figure(args.stage, frames, args.smoothing_window, args.max_step, teacher)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(f"{args.output_prefix}.{suffix}",
                       dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)

    if args.csv_output:
        merged = pd.concat([f.assign(stage=s["title"]) for s, f in zip(args.stage, frames)],
                           ignore_index=True)
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.csv_output, index=False)

    for stage, frame in zip(args.stage, frames):
        sm = smooth(frame["response_length/mean"].to_numpy(), args.smoothing_window)
        first, last = sm[~np.isnan(sm)][0], sm[~np.isnan(sm)][-1]
        peak = int(frame["step"].to_numpy()[int(np.nanargmax(sm))])
        print(f"{stage['title']:24s} steps {frame['step'].min()}-{frame['step'].max()}  "
              f"length {first:.0f} -> {last:.0f}  (peak {np.nanmax(sm):.0f} at {peak})  "
              f"final student endoftext {frame['student_eos_prob/token_1/last_token'].iloc[-1]:.4f}  "
              f"im_end {frame['student_eos_prob/token_250019/last_token'].iloc[-1]:.4f}")
    if teacher is not None:
        print(f"teacher reference: mean length {teacher['length']:.0f} tokens, "
              f"clip {100 * teacher['clip']:.1f}%")
    print(f"wrote {args.output_prefix}.png / .pdf")


if __name__ == "__main__":
    main()
