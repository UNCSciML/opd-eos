#!/usr/bin/env python3
"""Terminal-token probability at the last token, one row per model pair.

Every panel is one terminal token of that pair, carrying four curves: student and
teacher, under the no-fix baseline and under the semantic-class fix. Reading a row
left to right shows where each side puts its termination mass, and whether the two
sides ever agree on the same surface form.

Model pairs declare their own terminal tokens, so rows may have different panel
counts; unused cells are removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ARM_COLORS = {"baseline": "#E69F00", "semantic_class": "#009E73"}
ARM_LABELS = {"baseline": "No EOS fix", "semantic_class": "Semantic fix"}
ROLE_STYLES = {"student": {"linestyle": "-"}, "teacher": {"linestyle": (0, (4, 1.6))}}
LOG_FLOOR = 1e-9


def parse_row(value: str) -> dict:
    """Parse TITLE;BASELINE_CSV;SEMANTIC_CSV;ID:NAME,ID:NAME[,...].

    Semicolons, not pipes: terminal token names contain pipes (<|eot_id|>).
    """
    parts = [p.strip() for p in value.split(";")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--row must be TITLE;BASELINE_CSV;SEMANTIC_CSV;ID:NAME,ID:NAME")
    tokens = []
    for spec in parts[3].split(","):
        tid, _, name = spec.strip().partition(":")
        if not tid or not name:
            raise argparse.ArgumentTypeError(f"bad token spec {spec!r}; want ID:NAME")
        tokens.append({"id": tid.strip(), "name": name.strip()})
    return {"title": parts[0], "baseline": Path(parts[1]),
            "semantic_class": Path(parts[2]), "tokens": tokens}


def load_history(path: Path, arm: str, tokens: list[dict]) -> pd.DataFrame:
    history = pd.read_csv(path, low_memory=False)
    step_column = "training/global_step" if "training/global_step" in history else "_step"
    columns = {f"{role}_{t['id']}": f"{role}_eos_prob/token_{t['id']}/last_token"
               for role in ROLE_STYLES for t in tokens}
    missing = [c for c in columns.values() if c not in history]
    if missing or step_column not in history:
        raise ValueError(f"{path} is missing: {', '.join(missing) or step_column}")
    records = history[[step_column, *columns.values()]].rename(
        columns={step_column: "step", **{src: dst for dst, src in columns.items()}})
    records = records.apply(pd.to_numeric, errors="coerce").dropna()
    records = records.sort_values("step").drop_duplicates("step", keep="last")
    records["step"] = records["step"].astype(int)
    if records.empty:
        raise ValueError(f"{path} has no complete records")
    records.insert(0, "arm", arm)
    return records.reset_index(drop=True)


def wandb_ema(values: np.ndarray, smoothing_weight: float) -> np.ndarray:
    """Apply W&B-style debiased exponential moving-average smoothing."""
    values = np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values)
    last = 0.0
    for index, value in enumerate(values, start=1):
        last = last * smoothing_weight + (1.0 - smoothing_weight) * value
        smoothed[index - 1] = last / (1.0 - smoothing_weight**index)
    return smoothed


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)


def build_figure(rows: list[dict], frames: list[pd.DataFrame], smoothing: float,
                 max_step: int, log_scale: bool, raw_alpha: float) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ncols = max(len(r["tokens"]) for r in rows)
    figure, axes = plt.subplots(len(rows), ncols, figsize=(2.6 * ncols, 2.25 * len(rows)),
                                squeeze=False)
    panel = 0
    for r, (row, records) in enumerate(zip(rows, frames)):
        for c in range(ncols):
            axis = axes[r][c]
            if c >= len(row["tokens"]):
                axis.remove()
                continue
            token = row["tokens"][c]
            for arm in ARM_COLORS:
                group = records.query("arm == @arm").sort_values("step")
                if group.empty:
                    continue
                for role, role_style in ROLE_STYLES.items():
                    values = group[f"{role}_{token['id']}"].to_numpy(dtype=np.float64)
                    if log_scale:
                        values = np.maximum(values, LOG_FLOOR)
                        smoothed = np.exp(wandb_ema(np.log(values), smoothing))
                    else:
                        smoothed = wandb_ema(values, smoothing)
                    axis.plot(group["step"], values, color=ARM_COLORS[arm], linewidth=0.8,
                              alpha=raw_alpha, zorder=1, **role_style)
                    axis.plot(group["step"], smoothed, color=ARM_COLORS[arm], linewidth=1.5,
                              zorder=3, label=f"{role.capitalize()}, {ARM_LABELS[arm]}",
                              **role_style)
            axis.set_title(f"({'abcdefghi'[panel]}) {row['title']}\n"
                           f"$e_{c + 1}$ = {token['name']} (id {token['id']})", pad=5, y=1.0)
            _style(axis)
            if log_scale:
                axis.set_yscale("log")
                axis.set_ylim(LOG_FLOOR * 0.6, 2.0)
            else:
                axis.set_ylim(-0.03, 1.03)
            axis.set_xlim(0, max_step)
            axis.set_xticks(np.arange(40, max_step + 1, 40))
            if r == len(rows) - 1 or c >= len(rows[min(r + 1, len(rows) - 1)]["tokens"]):
                axis.set_xlabel("Training step")
            if c == 0:
                axis.set_ylabel("Probability at last token")
            panel += 1

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uniq.append((h, l))
    figure.legend([h for h, _ in uniq], [l for _, l in uniq], loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                  handlelength=2.2, columnspacing=1.4)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.845,
                           wspace=0.24, hspace=0.62)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", action="append", required=True, type=parse_row,
                        help="TITLE;BASELINE_CSV;SEMANTIC_CSV;ID:NAME,ID:NAME (repeatable)")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--smoothing-weight", type=float, default=0.8)
    parser.add_argument("--raw-alpha", type=float, default=0.30)
    parser.add_argument("--max-step", type=int, default=200)
    args = parser.parse_args()

    frames = []
    for row in args.row:
        frames.append(pd.concat(
            [load_history(row[arm], arm, row["tokens"]) for arm in ARM_COLORS],
            ignore_index=True))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for log_scale in (False, True):
        figure = build_figure(args.row, frames, args.smoothing_weight, args.max_step,
                              log_scale, args.raw_alpha)
        stem = f"{args.output_prefix}_{'log' if log_scale else 'linear'}"
        for suffix in ("png", "pdf"):
            figure.savefig(f"{stem}.{suffix}", dpi=600 if suffix == "png" else None,
                           bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {stem}.png / .pdf")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([f.assign(pair=r["title"]) for r, f in zip(args.row, frames)],
                  ignore_index=True).to_csv(args.csv_output, index=False)

    for row, records in zip(args.row, frames):
        print(f"\n{row['title']}")
        for arm in ARM_COLORS:
            g = records.query("arm == @arm").sort_values("step")
            final = g.iloc[-1]
            cells = "  ".join(
                f"{t['name']}: stu {final['student_' + t['id']]:.3g} / tch {final['teacher_' + t['id']]:.3g}"
                for t in row["tokens"])
            print(f"   {ARM_LABELS[arm]:14s} step {int(final['step'])}  {cells}")


if __name__ == "__main__":
    main()
