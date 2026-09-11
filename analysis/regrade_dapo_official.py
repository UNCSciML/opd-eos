#!/usr/bin/env python3
"""Regrade saved DAPO-prompt evaluations with the official DAPO scorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORER_PATH = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "math_dapo.py"


def _load_official_compute_score():
    spec = importlib.util.spec_from_file_location("opd_length_inflation_math_dapo", OFFICIAL_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official DAPO scorer from {OFFICIAL_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


OFFICIAL_COMPUTE_SCORE = _load_official_compute_score()
TASKS = ("AIME24", "AIME25", "AMC23")
TRAIN_TEMPLATES = ("ttrl", "dapo")


def score_response_official_dapo(response: str, ground_truth: str) -> bool:
    return bool(OFFICIAL_COMPUTE_SCORE(response, ground_truth)["acc"])


def regrade_eval_root(eval_root: Path, train_template: str) -> pd.DataFrame:
    eval_root = Path(eval_root)
    if train_template not in {"ttrl", "dapo"}:
        raise ValueError(f"Unsupported train template: {train_template}")

    config_path = eval_root / "eval_run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("prompt_template") != "dapo":
        raise ValueError(f"Expected a DAPO-prompt eval root, found {config.get('prompt_template')!r}: {eval_root}")

    summary_rows = []
    for output_path in sorted(eval_root.glob("step_*/*.jsonl")):
        step_match = re.fullmatch(r"step_(\d+)", output_path.parent.name)
        if step_match is None:
            continue
        task = output_path.name.split("_", maxsplit=1)[0].upper()
        grouped: dict[int, dict[str, object]] = {}
        with output_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                example_id = int(row["example_id"])
                answer = str(row["answer"])
                item = grouped.setdefault(example_id, {"answer": answer, "scores": []})
                if item["answer"] != answer:
                    raise ValueError(f"Inconsistent answers for example {example_id}: {output_path}")
                item["scores"].append(score_response_official_dapo(str(row["response"]), answer))

        if not grouped:
            raise ValueError(f"No responses in {output_path}")
        example_scores = [sum(item["scores"]) / len(item["scores"]) for item in grouped.values()]
        summary_rows.append(
            {
                "train_template": train_template,
                "eval_template": "dapo",
                "step": int(step_match.group(1)),
                "model_label": output_path.parent.name,
                "task": task,
                "mean_score": sum(example_scores) / len(example_scores),
                "best_score": sum(score > 0 for score in example_scores) / len(example_scores),
                "solve_none": sum(score == 0 for score in example_scores),
                "solve_all": sum(score == 1 for score in example_scores),
                "num_examples": len(grouped),
                "num_rollouts": sum(len(item["scores"]) for item in grouped.values()),
            }
        )

    if not summary_rows:
        raise ValueError(f"No step_*/*.jsonl files found in {eval_root}")
    return pd.DataFrame(summary_rows).sort_values(["step", "task"], ignore_index=True)


def parse_result(value: str) -> tuple[str, Path]:
    parts = value.split(",", maxsplit=1)
    if len(parts) != 2 or parts[0] not in TRAIN_TEMPLATES:
        raise argparse.ArgumentTypeError("--result must be TRAIN_TEMPLATE,EVAL_ROOT with TRAIN_TEMPLATE=ttrl or dapo")
    return parts[0], Path(parts[1])


def write_official_summary(eval_root: Path, rows: pd.DataFrame) -> None:
    rows.to_csv(eval_root / "grading_summary_official_dapo.csv", index=False)
    records = json.loads(rows.to_json(orient="records"))
    (eval_root / "grading_summary_official_dapo.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_task_rows(rows: pd.DataFrame, source: Path) -> None:
    for step, group in rows.groupby("step", sort=True):
        if len(group) != len(TASKS) or set(group["task"]) != set(TASKS):
            raise ValueError(f"{source} step {step} must contain exactly {TASKS}")


def build_macro_comparison(results: list[tuple[str, Path, pd.DataFrame]]) -> pd.DataFrame:
    combined = []
    for train_template, eval_root, official in results:
        semantic_path = eval_root / "grading_summary.csv"
        semantic = pd.read_csv(semantic_path)
        required = {"step", "task", "mean_score", "num_rollouts", "output_length_mean", "truncation_rate"}
        missing = required - set(semantic.columns)
        if missing:
            raise ValueError(f"{semantic_path} is missing columns: {sorted(missing)}")
        _validate_task_rows(official, eval_root / "grading_summary_official_dapo.csv")
        _validate_task_rows(semantic, semantic_path)

        official_macro = official.groupby("step", sort=True)["mean_score"].mean().rename("official_macro_accuracy")
        semantic_macro = semantic.groupby("step", sort=True)["mean_score"].mean().rename("semantic_macro_accuracy")
        for step in sorted(set(official_macro.index) | set(semantic_macro.index)):
            if step not in official_macro.index or step not in semantic_macro.index:
                raise ValueError(f"Official and semantic summaries have different steps in {eval_root}")
            step_rows = semantic[semantic["step"] == step]
            weights = step_rows["num_rollouts"].to_numpy(dtype=float)
            official_score = float(official_macro.loc[step])
            semantic_score = float(semantic_macro.loc[step])
            combined.append(
                {
                    "train_template": train_template,
                    "eval_template": "dapo",
                    "step": int(step),
                    "official_macro_accuracy": official_score,
                    "semantic_macro_accuracy": semantic_score,
                    "official_minus_semantic": official_score - semantic_score,
                    "output_length_mean": float(np.average(step_rows["output_length_mean"], weights=weights)),
                    "truncation_rate": float(np.average(step_rows["truncation_rate"], weights=weights)),
                }
            )
    return pd.DataFrame(combined).sort_values(["train_template", "step"], ignore_index=True)


def summarize(comparison: pd.DataFrame) -> dict:
    conditions = {}
    for train_template, group in comparison.groupby("train_template", sort=True):
        ordered = group.sort_values("step")
        final = ordered.iloc[-1]
        best_index = ordered["official_macro_accuracy"].idxmax()
        best = ordered.loc[best_index]
        conditions[f"{train_template}_train__dapo_eval"] = {
            "final_step": int(final["step"]),
            "final_official_macro_accuracy": float(final["official_macro_accuracy"]),
            "final_semantic_macro_accuracy": float(final["semantic_macro_accuracy"]),
            "final_official_minus_semantic": float(final["official_minus_semantic"]),
            "best_official_step": int(best["step"]),
            "best_official_macro_accuracy": float(best["official_macro_accuracy"]),
            "mean_official_minus_semantic_across_steps": float(ordered["official_minus_semantic"].mean()),
        }
    return {
        "score_aggregation": "unweighted mean of AIME24, AIME25, and AMC23 mean_score",
        "official_scorer": "verl/verl/utils/reward_score/math_dapo.py (final 300 characters, Answer: extraction)",
        "generation": "reused saved @16 responses; no regeneration",
        "conditions": conditions,
    }


def create_figure(comparison: pd.DataFrame, output_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"ttrl": "#0072B2", "dapo": "#D55E00"}
    fig, (score_ax, gap_ax) = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="white")
    for train_template in TRAIN_TEMPLATES:
        group = comparison[comparison["train_template"] == train_template].sort_values("step")
        if group.empty:
            continue
        label_prefix = f"Train {train_template.upper()} → Eval DAPO"
        score_ax.plot(
            group["step"],
            100 * group["semantic_macro_accuracy"],
            color=colors[train_template],
            marker="o",
            linestyle="--",
            linewidth=1.7,
            label=f"{label_prefix}: semantic",
        )
        score_ax.plot(
            group["step"],
            100 * group["official_macro_accuracy"],
            color=colors[train_template],
            marker="s",
            linewidth=2.1,
            label=f"{label_prefix}: official",
        )
        gap_ax.plot(
            group["step"],
            100 * group["official_minus_semantic"],
            color=colors[train_template],
            marker="s",
            linewidth=2.1,
            label=f"Train {train_template.upper()}",
        )

    for axis in (score_ax, gap_ax):
        axis.set_xlabel("Training checkpoint step")
        axis.grid(axis="y", alpha=0.25)
    score_ax.set_title("DAPO-prompt accuracy")
    score_ax.set_ylabel("Macro accuracy (%)")
    score_ax.legend(frameon=False, fontsize=8)
    gap_ax.set_title("Penalty from official DAPO scorer")
    gap_ax.set_ylabel("Official − semantic (percentage points)")
    gap_ax.axhline(0, color="#777777", linestyle=":", linewidth=1)
    gap_ax.legend(frameon=False)
    fig.suptitle("Baseline OPD: semantic vs official DAPO scoring", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output_prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_result,
        required=True,
        metavar="TRAIN_TEMPLATE,EVAL_ROOT",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_template = {train_template: eval_root for train_template, eval_root in args.result}
    if len(args.result) != len(TRAIN_TEMPLATES) or set(by_template) != set(TRAIN_TEMPLATES):
        raise ValueError("Provide the ttrl and dapo train-template DAPO eval roots exactly once")

    results = []
    for train_template in TRAIN_TEMPLATES:
        eval_root = by_template[train_template]
        official = regrade_eval_root(eval_root, train_template)
        write_official_summary(eval_root, official)
        results.append((train_template, eval_root, official))

    comparison = build_macro_comparison(results)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summarize(comparison), indent=2) + "\n",
        encoding="utf-8",
    )
    create_figure(comparison, args.output_prefix)
    print(comparison.to_string(index=False))
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.json')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
