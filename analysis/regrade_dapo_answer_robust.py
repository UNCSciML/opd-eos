#!/usr/bin/env python3
"""Regrade DAPO outputs with robust math delimiters after ``Answer:``."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SCORER_PATH = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "math_dapo.py"
TAIL_CHARACTER_LIMIT = 300
ANSWER_MARKER = re.compile(r"(?i)Answer\s*:\s*")


def _load_official_scorer_module():
    spec = importlib.util.spec_from_file_location("opd_length_inflation_math_dapo_robust", OFFICIAL_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official DAPO scorer from {OFFICIAL_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL_SCORER = _load_official_scorer_module()
OFFICIAL_COMPUTE_SCORE = OFFICIAL_SCORER.compute_score
OFFICIAL_NORMALIZE_FINAL_ANSWER = OFFICIAL_SCORER.normalize_final_answer


def _extract_immediate_answer_block(text: str) -> str | None:
    payload = text.lstrip()
    if not payload:
        return None

    delimiter = "$$" if payload.startswith("$$") else "$" if payload.startswith("$") else None
    if delimiter is None:
        return payload.split("\n", maxsplit=1)[0].strip()

    closing_index = payload.find(delimiter, len(delimiter))
    if closing_index < 0:
        return None
    same_line_remainder = payload[closing_index + len(delimiter) :].split("\n", maxsplit=1)[0]
    if same_line_remainder.strip():
        return None
    return payload[len(delimiter) : closing_index].strip()


def score_response_dapo_answer_robust(response: str, ground_truth: str) -> bool:
    """Apply official normalization after robustly unwrapping an ``Answer:`` value."""
    if bool(OFFICIAL_COMPUTE_SCORE(response, ground_truth)["acc"]):
        return True

    tail = response[-TAIL_CHARACTER_LIMIT:]
    matches = list(ANSWER_MARKER.finditer(tail))
    if not matches:
        return False

    answer_block = _extract_immediate_answer_block(tail[matches[-1].end() :])
    if answer_block is None:
        return False

    prediction = OFFICIAL_NORMALIZE_FINAL_ANSWER(answer_block)
    answer = OFFICIAL_NORMALIZE_FINAL_ANSWER(ground_truth)
    return prediction == answer


def regrade_eval_root(eval_root: Path) -> list[dict[str, object]]:
    eval_root = Path(eval_root)
    config = json.loads((eval_root / "eval_run_config.json").read_text(encoding="utf-8"))
    if config.get("prompt_template") != "dapo":
        raise ValueError(f"Expected a DAPO-prompt eval root, found {config.get('prompt_template')!r}: {eval_root}")

    summary_rows: list[dict[str, object]] = []
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
                ground_truth = str(row["answer"])
                item = grouped.setdefault(example_id, {"answer": ground_truth, "scores": []})
                if item["answer"] != ground_truth:
                    raise ValueError(f"Inconsistent answers for example {example_id}: {output_path}")
                scores = item["scores"]
                assert isinstance(scores, list)
                scores.append(score_response_dapo_answer_robust(str(row["response"]), ground_truth))

        if not grouped:
            raise ValueError(f"No responses in {output_path}")
        example_scores = []
        num_rollouts = 0
        for item in grouped.values():
            scores = item["scores"]
            assert isinstance(scores, list)
            example_scores.append(sum(scores) / len(scores))
            num_rollouts += len(scores)
        summary_rows.append(
            {
                "step": int(step_match.group(1)),
                "model_label": output_path.parent.name,
                "task": task,
                "mean_score": sum(example_scores) / len(example_scores),
                "best_score": sum(score > 0 for score in example_scores) / len(example_scores),
                "solve_none": sum(score == 0 for score in example_scores),
                "solve_all": sum(score == 1 for score in example_scores),
                "num_examples": len(grouped),
                "num_rollouts": num_rollouts,
            }
        )

    if not summary_rows:
        raise ValueError(f"No step_*/*.jsonl files found in {eval_root}")
    return sorted(summary_rows, key=lambda row: (int(row["step"]), str(row["task"])))


def write_robust_summary(eval_root: Path, rows: list[dict[str, object]]) -> None:
    json_path = eval_root / "grading_summary_dapo_answer_robust.json"
    csv_path = eval_root / "grading_summary_dapo_answer_robust.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for eval_root in args.eval_root:
        rows = regrade_eval_root(eval_root)
        write_robust_summary(eval_root, rows)
        macro_by_step: dict[int, list[float]] = {}
        for row in rows:
            macro_by_step.setdefault(int(row["step"]), []).append(float(row["mean_score"]))
        for step, task_scores in sorted(macro_by_step.items()):
            print(f"{eval_root} step={step} macro={sum(task_scores) / len(task_scores):.8f}")


if __name__ == "__main__":
    main()
