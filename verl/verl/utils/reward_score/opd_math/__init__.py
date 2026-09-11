"""Format-agnostic math accuracy logging for OPD template experiments."""

from __future__ import annotations

import re
import traceback

from verl.utils.reward_score.ttrl_math.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)


def extract_answer(response: str) -> str | None:
    if "\\boxed" in response:
        return extract_boxed_answer(response)
    matches = re.findall(r"(?im)^\s*Answer:\s*(.+?)\s*$", response)
    if not matches:
        return None
    answer = matches[-1].strip()
    if len(answer) >= 2 and answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    return answer


def compute_score(model_response, gt_answer):
    model_answer = extract_answer(model_response)
    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
    gt_answer = str(gt_answer)
    is_correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    return {
        "score": float(is_correct),
        "format_score": 1.0,
        "acc": bool(is_correct),
        "extracted_gt": gt_answer,
        "pred": model_answer,
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    del data_source, extra_info, sandbox_fusion_url, concurrent_semaphore
    try:
        return compute_score(solution_str, ground_truth)
    except Exception as exc:
        print(f"[ERROR] OPD math grading failed: {exc}")
        traceback.print_exc()
        raise
