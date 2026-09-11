#!/usr/bin/env python3
"""Plot teacher EOS probabilities at positions where the student sampled E1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARRAY_COLUMNS = (
    "response_length",
    "teacher_prob_e1",
    "teacher_prob_e2",
    "teacher_eos_mass",
    "teacher_e1_share",
    "teacher_e2_share",
    "logprob_e2_minus_e1",
)


def load_records(input_dir: Path) -> tuple[pd.DataFrame, tuple[int, int]]:
    files = sorted(input_dir.glob("step_*_teacher_eos_at_student_eos.npz"))
    if not files:
        raise FileNotFoundError(f"No diagnostic NPZ files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    eos_token_ids: tuple[int, int] | None = None
    for path in files:
        with np.load(path) as data:
            missing = set(ARRAY_COLUMNS) - set(data.files)
            if missing:
                raise ValueError(f"{path.name} is missing arrays: {sorted(missing)}")
            current_ids = tuple(int(value) for value in np.asarray(data["eos_token_ids"]).tolist())
            if len(current_ids) != 2:
                raise ValueError(f"{path.name} must contain exactly two EOS token IDs")
            if eos_token_ids is not None and current_ids != eos_token_ids:
                raise ValueError("EOS token IDs differ across diagnostic files")
            eos_token_ids = current_ids

            step = int(np.asarray(data["global_step"]).item())
            arrays = {name: np.asarray(data[name]) for name in ARRAY_COLUMNS}
            lengths = {len(values) for values in arrays.values()}
            if len(lengths) != 1:
                raise ValueError(f"{path.name} contains arrays of inconsistent lengths")
            frame = pd.DataFrame(arrays)
            frame.insert(0, "sample_index", np.arange(len(frame), dtype=int))
            frame.insert(0, "global_step", step)
            frames.append(frame)

    assert eos_token_ids is not None
    return pd.concat(frames, ignore_index=True), eos_token_ids


def summarize(records: pd.DataFrame, eos_token_ids: tuple[int, int]) -> dict[str, float | int | list[int]]:
    log_gap = records["logprob_e2_minus_e1"]
    return {
        "count": int(len(records)),
        "global_steps": sorted(int(step) for step in records["global_step"].unique()),
        "eos_token_ids": list(eos_token_ids),
        "e2_probability_greater_fraction": float(
            (records["teacher_prob_e2"] > records["teacher_prob_e1"]).mean()
        ),
        "teacher_e1_share_mean": float(records["teacher_e1_share"].mean()),
        "teacher_e2_share_mean": float(records["teacher_e2_share"].mean()),
        "teacher_e2_share_median": float(records["teacher_e2_share"].median()),
        "logprob_e2_minus_e1_mean": float(log_gap.mean()),
        "logprob_e2_minus_e1_median": float(log_gap.median()),
        "e2_to_e1_geometric_mean_ratio": float(np.exp(log_gap.mean())),
    }


def _relative_probability_density(probabilities: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Boundary-corrected Gaussian KDE, normalized to unit peak height."""
    values = np.asarray(probabilities, dtype=float)
    scale = min(float(values.std(ddof=1)), float(np.subtract(*np.percentile(values, [75, 25]))) / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(values.std(ddof=1))
    bandwidth = float(np.clip(0.9 * scale * len(values) ** (-0.2), 0.035, 0.08))
    reflected = np.concatenate((values, -values, 2.0 - values))
    standardized = (grid[:, None] - reflected[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized**2).sum(axis=1)
    return density / density.max()


def create_figure(
    records: pd.DataFrame,
    _summary: dict[str, float | int | list[int]],
    _eos_token_ids: tuple[int, int],
    output_prefix: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.0, 3.35), facecolor="white")
    grid = np.linspace(0.0, 1.0, 1000)
    distributions = (
        ("teacher_prob_e1", "Teacher P(<|endoftext|>)", "#0072B2", "-"),
        ("teacher_prob_e2", "Teacher P(<|im_end|>)", "#D55E00", "--"),
    )
    for metric, label, color, linestyle in distributions:
        density = _relative_probability_density(records[metric].to_numpy(), grid)
        ax.fill_between(grid, density, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(
            grid,
            density,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Teacher probability at the student stopping position")
    ax.set_ylabel("Relative density")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.8)
    ax.legend(loc="upper center", ncol=2, frameon=False, handlelength=2.6, columnspacing=1.5)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.96)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, eos_token_ids = load_records(args.input_dir)
    summary = summarize(records, eos_token_ids)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    create_figure(records, summary, eos_token_ids, args.output_prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
