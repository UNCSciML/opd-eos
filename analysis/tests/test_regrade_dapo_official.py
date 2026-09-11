import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from analysis.regrade_dapo_official import regrade_eval_root, score_response_official_dapo


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "regrade_dapo_official.py"


def test_official_dapo_scorer_only_sees_the_final_300_characters() -> None:
    trailing_junk = "x" * 301

    assert score_response_official_dapo(f"work\nAnswer: 4\n{trailing_junk}", "4") is False
    assert score_response_official_dapo(f"{trailing_junk}\nAnswer: 4", "4") is True


def test_regrade_eval_root_aggregates_rollouts_by_example(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    step_dir = eval_root / "step_0020"
    step_dir.mkdir(parents=True)
    (eval_root / "eval_run_config.json").write_text(
        json.dumps({"prompt_template": "dapo", "n": 2, "max_tokens": 8192}),
        encoding="utf-8",
    )
    rows = [
        {"example_id": 0, "answer": "4", "response": "work\nAnswer: 4"},
        {"example_id": 0, "answer": "4", "response": f"Answer: 4\n{'x' * 301}"},
        {"example_id": 1, "answer": "5", "response": "work\nAnswer: 5"},
        {"example_id": 1, "answer": "5", "response": "again\nAnswer: 5"},
    ]
    output = step_dir / "aime24_t0.7_p0.95_n2-MNT8192.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = regrade_eval_root(eval_root, train_template="ttrl")

    assert result.to_dict("records") == [
        {
            "train_template": "ttrl",
            "eval_template": "dapo",
            "step": 20,
            "model_label": "step_0020",
            "task": "AIME24",
            "mean_score": pytest.approx(0.75),
            "best_score": pytest.approx(1.0),
            "solve_none": 0,
            "solve_all": 1,
            "num_examples": 2,
            "num_rollouts": 4,
        }
    ]


def test_cli_preserves_semantic_summary_and_writes_official_macro_comparison(tmp_path: Path) -> None:
    result_args = []
    official_correctness = {
        "ttrl": {"AIME24": True, "AIME25": True, "AMC23": False},
        "dapo": {"AIME24": True, "AIME25": False, "AMC23": False},
    }
    roots = {}
    for train_template in ("ttrl", "dapo"):
        eval_root = tmp_path / f"{train_template}-train"
        roots[train_template] = eval_root
        step_dir = eval_root / "step_0020"
        step_dir.mkdir(parents=True)
        (eval_root / "eval_run_config.json").write_text(
            json.dumps({"prompt_template": "dapo", "n": 1, "max_tokens": 8192}),
            encoding="utf-8",
        )
        semantic_rows = []
        for task, is_correct in official_correctness[train_template].items():
            response = "work\nAnswer: 4" if is_correct else "work\nAnswer: 7"
            output = step_dir / f"{task.lower()}_t0.7_p0.95_n1-MNT8192.jsonl"
            output.write_text(
                json.dumps({"example_id": 0, "answer": "4", "response": response}) + "\n",
                encoding="utf-8",
            )
            semantic_rows.append(
                {
                    "step": 20,
                    "task": task,
                    "mean_score": 0.5,
                    "num_rollouts": 1,
                    "output_length_mean": 100.0,
                    "truncation_rate": 0.25,
                }
            )
        pd.DataFrame(semantic_rows).to_csv(eval_root / "grading_summary.csv", index=False)
        (eval_root / "grading_summary.json").write_text("original semantic summary\n", encoding="utf-8")
        result_args.extend(["--result", f"{train_template},{eval_root}"])

    output_prefix = tmp_path / "combined" / "baseline-dapo-official"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *result_args, "--output-prefix", str(output_prefix)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for eval_root in roots.values():
        assert (eval_root / "grading_summary.json").read_text(encoding="utf-8") == "original semantic summary\n"
        assert (eval_root / "grading_summary_official_dapo.csv").is_file()
        assert (eval_root / "grading_summary_official_dapo.json").is_file()

    comparison = pd.read_csv(output_prefix.with_suffix(".csv")).sort_values("train_template")
    dapo = comparison.query("train_template == 'dapo'").iloc[0]
    ttrl = comparison.query("train_template == 'ttrl'").iloc[0]
    assert dapo["official_macro_accuracy"] == pytest.approx(1 / 3)
    assert ttrl["official_macro_accuracy"] == pytest.approx(2 / 3)
    assert dapo["semantic_macro_accuracy"] == pytest.approx(0.5)
    assert ttrl["semantic_macro_accuracy"] == pytest.approx(0.5)
    assert dapo["official_minus_semantic"] == pytest.approx(-1 / 6)
    assert ttrl["official_minus_semantic"] == pytest.approx(1 / 6)
    assert output_prefix.with_suffix(".json").is_file()
    assert output_prefix.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert output_prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
