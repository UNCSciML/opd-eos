#!/usr/bin/env python3
"""Plot no-fix baseline curves and a teacher reference under the original DAPO grader."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Callable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORER_PATH = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "math_dapo.py"
ADAPTED_SCORER_PATH = REPO_ROOT / "analysis" / "regrade_dapo_answer_robust.py"
TASKS = ("AIME24", "AIME25", "AMC23")
TRAIN_TEMPLATES = ("ttrl", "dapo")
STYLES = {
    "ttrl": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "dapo": {"color": "#D55E00", "marker": "^", "linestyle": "--"},
}


def _load_official_compute_score():
    spec = importlib.util.spec_from_file_location("opd_length_inflation_math_dapo_for_plot", OFFICIAL_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load original DAPO scorer from {OFFICIAL_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


OFFICIAL_COMPUTE_SCORE = _load_official_compute_score()


def _load_adapted_score_response():
    spec = importlib.util.spec_from_file_location("opd_length_inflation_answer_robust_for_plot", ADAPTED_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load adapted DAPO scorer from {ADAPTED_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_response_dapo_answer_robust


ADAPTED_SCORE_RESPONSE = _load_adapted_score_response()


def load_baseline_curves(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path)
    required = {"train_template", "eval_template", "step", "macro_accuracy"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    records = records[
        (records["eval_template"] == "dapo") & records["train_template"].isin(TRAIN_TEMPLATES)
    ].copy()
    if records.empty:
        raise ValueError(f"{path} contains no TTRL/DAPO-trained baseline rows evaluated with DAPO")

    expected_steps: tuple[int, ...] | None = None
    for train_template in TRAIN_TEMPLATES:
        group = records[records["train_template"] == train_template]
        if group.empty:
            raise ValueError(f"{path} has no {train_template.upper()}-trained DAPO-eval curve")
        if group["step"].duplicated().any():
            raise ValueError(f"{path} has duplicate steps for {train_template.upper()}-trained DAPO eval")
        steps = tuple(sorted(int(step) for step in group["step"]))
        if 0 not in steps:
            raise ValueError(f"{path} has no step 0 for {train_template.upper()}-trained DAPO eval")
        if expected_steps is None:
            expected_steps = steps
        elif steps != expected_steps:
            raise ValueError(f"{path} has different checkpoint steps for the two training templates")

    records["step"] = records["step"].astype(int)
    records["macro_accuracy"] = records["macro_accuracy"].astype(float)
    return records.sort_values(["train_template", "step"], ignore_index=True)


def regrade_teacher(
    eval_root: Path,
    score_response: Callable[[str, str], bool],
) -> tuple[float, dict[str, float]]:
    config_path = eval_root / "eval_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("prompt_template") != "dapo":
        raise ValueError(
            f"Expected a DAPO-prompt teacher eval root, found {config.get('prompt_template')!r}: {eval_root}"
        )

    jsonl_paths = sorted(eval_root.glob("step_*/*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"No step_*/*.jsonl files found in {eval_root}")
    step_dirs = {path.parent.name for path in jsonl_paths}
    if step_dirs != {"step_0000"}:
        raise ValueError(f"Teacher reference must contain only step_0000, found {sorted(step_dirs)}")

    task_scores: dict[str, float] = {}
    for output_path in jsonl_paths:
        task = output_path.name.split("_", maxsplit=1)[0].upper()
        if task not in TASKS:
            continue
        if task in task_scores:
            raise ValueError(f"Multiple teacher JSONL files found for {task}: {eval_root}")

        grouped: dict[int, dict[str, object]] = {}
        with output_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                example_id = int(row["example_id"])
                answer = str(row["answer"])
                item = grouped.setdefault(example_id, {"answer": answer, "scores": []})
                if item["answer"] != answer:
                    raise ValueError(f"Inconsistent answers for example {example_id}: {output_path}")
                item["scores"].append(score_response(str(row["response"]), answer))

        if not grouped:
            raise ValueError(f"No responses in {output_path}")
        example_scores = [sum(item["scores"]) / len(item["scores"]) for item in grouped.values()]
        task_scores[task] = sum(example_scores) / len(example_scores)

    if set(task_scores) != set(TASKS):
        raise ValueError(f"Teacher root must contain exactly {TASKS}; found {sorted(task_scores)}")
    macro_accuracy = sum(task_scores[task] for task in TASKS) / len(TASKS)
    return macro_accuracy, task_scores


def build_export(
    baselines: dict[str, pd.DataFrame],
    teacher_results: dict[str, tuple[float, dict[str, float]]],
    baseline_sources: dict[str, Path],
    teacher_source: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = {"ttrl": "TTRL-trained baseline", "dapo": "DAPO-trained baseline"}
    for grader, baseline in baselines.items():
        for record in baseline.to_dict(orient="records"):
            rows.append(
                {
                    "series_type": "baseline",
                    "series_label": labels[str(record["train_template"])],
                    "train_template": record["train_template"],
                    "eval_template": "dapo",
                    "step": int(record["step"]),
                    "macro_accuracy": float(record["macro_accuracy"]),
                    "aime24_accuracy": None,
                    "aime25_accuracy": None,
                    "amc23_accuracy": None,
                    "grader": grader,
                    "source": str(baseline_sources[grader].resolve()),
                }
            )

        teacher_macro, teacher_task_scores = teacher_results[grader]
        step_min = int(baseline["step"].min())
        step_max = int(baseline["step"].max())
        for step in (step_min, step_max):
            rows.append(
                {
                    "series_type": "teacher_reference",
                    "series_label": "Qwen3-4B teacher",
                    "train_template": "teacher",
                    "eval_template": "dapo",
                    "step": step,
                    "macro_accuracy": teacher_macro,
                    "aime24_accuracy": teacher_task_scores["AIME24"],
                    "aime25_accuracy": teacher_task_scores["AIME25"],
                    "amc23_accuracy": teacher_task_scores["AMC23"],
                    "grader": grader,
                    "source": str(teacher_source.resolve()),
                }
            )
    return pd.DataFrame(rows)


def create_figure(records: pd.DataFrame, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), facecolor="white", sharey=True)
    all_scores = 100 * records["macro_accuracy"].astype(float)
    score_range = max(1.0, float(all_scores.max() - all_scores.min()))
    y_limits = (
        max(0.0, float(all_scores.min()) - 0.12 * score_range),
        float(all_scores.max()) + 0.16 * score_range,
    )
    step_min = int(records["step"].min())
    step_max = int(records["step"].max())
    grader_panels = (
        ("original_official_dapo", "Original DAPO Avg@16 (%)"),
        ("answer_robust_dapo", "Adapted DAPO Avg@16 (%)"),
    )
    legend_handles = []
    for ax, (grader, ylabel) in zip(axes, grader_panels):
        panel = records[records["grader"] == grader]
        baseline = panel[panel["series_type"] == "baseline"]
        panel_handles = []
        for train_template in TRAIN_TEMPLATES:
            group = baseline[baseline["train_template"] == train_template].sort_values("step")
            style = STYLES[train_template]
            line = ax.plot(
                group["step"],
                100 * group["macro_accuracy"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.8,
                linewidth=2.0,
                label=str(group.iloc[0]["series_label"]),
            )[0]
            panel_handles.append(line)

        teacher = panel[panel["series_type"] == "teacher_reference"].iloc[0]
        teacher_score = 100 * float(teacher["macro_accuracy"])
        teacher_line = ax.axhline(
            teacher_score,
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.7,
            label=str(teacher["series_label"]),
            zorder=1,
        )
        ax.annotate(
            f"{teacher_score:.2f}%",
            xy=(step_max, teacher_score),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="#555555",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
        )
        if not legend_handles:
            legend_handles = [*panel_handles, teacher_line]

        ax.set_ylim(*y_limits)
        ax.set_xlim(step_min - 5, step_max + 5)
        if step_min == 0 and step_max == 200:
            ax.set_xticks([0, 20, 80, 140, 200])
        ax.set_xlabel("Training checkpoint step")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.7, alpha=0.75)

    fig.legend(
        legend_handles,
        ["TTRL-trained baseline", "DAPO-trained baseline", "Qwen3-4B teacher"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=2.6,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.84, wspace=0.18)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="CSV containing no-fix baseline scores from the original official DAPO grader.",
    )
    parser.add_argument(
        "--teacher-eval-root",
        type=Path,
        required=True,
        help="Saved Qwen3-4B DAPO-prompt @16 responses to regrade with the same scorer.",
    )
    parser.add_argument(
        "--adapted-baseline-csv",
        type=Path,
        required=True,
        help="CSV containing the same no-fix baseline curves under the adapted DAPO grader.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = {
        "original_official_dapo": load_baseline_curves(args.baseline_csv),
        "answer_robust_dapo": load_baseline_curves(args.adapted_baseline_csv),
    }
    original_steps = baselines["original_official_dapo"][["train_template", "step"]].reset_index(drop=True)
    adapted_steps = baselines["answer_robust_dapo"][["train_template", "step"]].reset_index(drop=True)
    if not original_steps.equals(adapted_steps):
        raise ValueError("Original and adapted baseline CSVs must contain the same training checkpoints")

    teacher_results = {
        "original_official_dapo": regrade_teacher(
            args.teacher_eval_root,
            lambda response, answer: bool(OFFICIAL_COMPUTE_SCORE(response, answer)["acc"]),
        ),
        "answer_robust_dapo": regrade_teacher(args.teacher_eval_root, ADAPTED_SCORE_RESPONSE),
    }
    exported = build_export(
        baselines,
        teacher_results,
        {
            "original_official_dapo": args.baseline_csv,
            "answer_robust_dapo": args.adapted_baseline_csv,
        },
        args.teacher_eval_root,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    create_figure(exported, args.output_prefix)
    for grader, (teacher_macro, teacher_task_scores) in teacher_results.items():
        print(f"Teacher {grader} macro accuracy: {100 * teacher_macro:.6f}%")
        for task in TASKS:
            print(f"  {task}: {100 * teacher_task_scores[task]:.6f}%")
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
