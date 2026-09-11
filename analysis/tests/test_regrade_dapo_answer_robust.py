import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "regrade_dapo_answer_robust.py"


def _score(response: str, answer: str) -> bool:
    scorer = importlib.import_module("analysis.regrade_dapo_answer_robust")
    return scorer.score_response_dapo_answer_robust(response, answer)


def test_accepts_double_dollar_box_immediately_after_answer_marker() -> None:
    assert _score(r"Answer: $$\boxed{204}$$", "204") is True


@pytest.mark.parametrize(
    "response",
    [
        "Answer: 204",
        "Answer:\n204",
        "Answer: $204$",
        r"Answer: \boxed{204}",
        "Answer:\n" + r"$$\boxed{204}$$",
        "Answer:\n$$\n" + r"\boxed{204}" + "\n$$",
    ],
)
def test_accepts_supported_answer_block_layouts(response: str) -> None:
    assert _score(response, "204") is True


@pytest.mark.parametrize(
    "response",
    [
        r"\boxed{204}",
        r"$$\boxed{204}$$",
        "Answer：204",
        "Answer: explanation\n" + r"\boxed{204}",
        "Answer: 205",
    ],
)
def test_rejects_missing_or_non_immediate_valid_answer(response: str) -> None:
    assert _score(response, "204") is False


def test_only_sees_the_final_300_characters() -> None:
    assert _score("Answer: 204\n" + "x" * 301, "204") is False
    assert _score("x" * 301 + "\nAnswer: 204", "204") is True


def test_uses_the_last_answer_marker() -> None:
    assert _score("Answer: 205\nAnswer: 204", "204") is True
    assert _score("Answer: 204\nAnswer: 205", "204") is False


def test_preserves_answers_already_accepted_by_the_official_scorer() -> None:
    assert _score("Answer: $25$ ✅", "25") is True


def test_cli_writes_separate_summary_without_overwriting_existing_scores(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    step_dir = eval_root / "step_0020"
    step_dir.mkdir(parents=True)
    (eval_root / "eval_run_config.json").write_text(
        json.dumps({"prompt_template": "dapo", "n": 2}),
        encoding="utf-8",
    )
    rows = [
        {"example_id": 0, "answer": "204", "response": r"Answer: $$\boxed{204}$$"},
        {"example_id": 0, "answer": "204", "response": r"\boxed{204}"},
    ]
    output = step_dir / "aime24_t0.7_p0.95_n2-MNT8192.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    official_path = eval_root / "grading_summary_official_dapo.json"
    semantic_path = eval_root / "grading_summary.json"
    official_path.write_text("official stays unchanged\n", encoding="utf-8")
    semantic_path.write_text("semantic stays unchanged\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval-root", str(eval_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert official_path.read_text(encoding="utf-8") == "official stays unchanged\n"
    assert semantic_path.read_text(encoding="utf-8") == "semantic stays unchanged\n"
    summary = json.loads((eval_root / "grading_summary_dapo_answer_robust.json").read_text(encoding="utf-8"))
    assert summary == [
        {
            "step": 20,
            "model_label": "step_0020",
            "task": "AIME24",
            "mean_score": 0.5,
            "best_score": 1.0,
            "solve_none": 0,
            "solve_all": 0,
            "num_examples": 1,
            "num_rollouts": 2,
        }
    ]
