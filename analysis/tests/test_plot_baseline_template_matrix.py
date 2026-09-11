import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_baseline_template_matrix.py"


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "grading_summary.csv", index=False)


def test_cli_uses_macro_accuracy_and_rollout_weighted_length(tmp_path: Path) -> None:
    """Catch task-size weighting of accuracy or unweighted aggregation of rollout length."""
    tasks = ["AIME24", "AIME25", "AMC23"]
    result_args: list[str] = []
    for condition_index, (train, evaluate) in enumerate(
        [("ttrl", "ttrl"), ("ttrl", "dapo"), ("dapo", "ttrl"), ("dapo", "dapo")]
    ):
        result_dir = tmp_path / f"{train}-{evaluate}"
        rows = []
        for step, score_shift in [(20, 0.00), (40, 0.03)]:
            for task, score, length, rollouts in zip(
                tasks,
                [0.10, 0.20, 0.30],
                [100.0, 200.0, 400.0],
                [10, 10, 20],
            ):
                rows.append(
                    {
                        "step": step,
                        "task": task,
                        "mean_score": score + score_shift + condition_index * 0.01,
                        "output_length_mean": length + step,
                        "truncation_rate": [0.1, 0.2, 0.4][tasks.index(task)],
                        "num_rollouts": rollouts,
                    }
                )
        _write_summary(result_dir, rows)
        result_args.extend(["--result", f"{train},{evaluate},{result_dir}"])

    output_prefix = tmp_path / "figure" / "baseline-template-matrix"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *result_args,
            "--paper-style",
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    first = exported.query("train_template == 'ttrl' and eval_template == 'ttrl' and step == 20").iloc[0]
    assert first["macro_accuracy"] == 0.20
    assert first["output_length_mean"] == 295.0
    assert first["truncation_rate"] == 0.275

    summary = json.loads(output_prefix.with_suffix(".json").read_text())
    ttrl_ttrl = summary["conditions"]["ttrl_train__ttrl_eval"]
    assert ttrl_ttrl["final_step"] == 40
    assert abs(ttrl_ttrl["final_macro_accuracy"] - 0.23) < 1e-12
    assert ttrl_ttrl["best_step"] == 40
    assert abs(ttrl_ttrl["best_macro_accuracy"] - 0.23) < 1e-12

    assert output_prefix.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert output_prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with Image.open(output_prefix.with_suffix(".png")) as image:
        assert image.width >= 3800
        assert 2.3 < image.width / image.height < 2.7


def test_cli_can_use_official_dapo_scores_without_changing_length_metrics(tmp_path: Path) -> None:
    """Catch plotting semantic scores for DAPO evals when official scoring is requested."""
    tasks = ["AIME24", "AIME25", "AMC23"]
    result_args: list[str] = []
    for train, evaluate in [("ttrl", "ttrl"), ("ttrl", "dapo"), ("dapo", "ttrl"), ("dapo", "dapo")]:
        result_dir = tmp_path / f"{train}-{evaluate}"
        result_dir.mkdir()
        semantic_rows = [
            {
                "step": 20,
                "task": task,
                "mean_score": 0.50,
                "output_length_mean": length,
                "truncation_rate": truncation,
                "num_rollouts": rollouts,
            }
            for task, length, truncation, rollouts in zip(
                tasks,
                [100.0, 200.0, 400.0],
                [0.10, 0.20, 0.40],
                [10, 10, 20],
            )
        ]
        pd.DataFrame(semantic_rows).to_csv(result_dir / "grading_summary.csv", index=False)
        if evaluate == "dapo":
            official_rows = [
                {
                    "step": 20,
                    "task": task,
                    "mean_score": score,
                    "num_rollouts": rollouts,
                }
                for task, score, rollouts in zip(tasks, [0.10, 0.20, 0.30], [10, 10, 20])
            ]
            pd.DataFrame(official_rows).to_csv(
                result_dir / "grading_summary_official_dapo.csv",
                index=False,
            )
        result_args.extend(["--result", f"{train},{evaluate},{result_dir}"])

    output_prefix = tmp_path / "figure" / "official-dapo"
    step_zero_csv = tmp_path / "step-zero.csv"
    pd.DataFrame(
        [
            {
                "eval_template": "ttrl",
                "step": 0,
                "macro_accuracy": 0.04,
                "output_length_mean": 400.0,
                "truncation_rate": 0.40,
            },
            {
                "eval_template": "dapo",
                "step": 0,
                "macro_accuracy": 0.03,
                "output_length_mean": 300.0,
                "truncation_rate": 0.30,
            },
        ]
    ).to_csv(step_zero_csv, index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *result_args,
            "--official-dapo",
            "--step-zero-csv",
            str(step_zero_csv),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    ttrl_dapo = exported.query(
        "train_template == 'ttrl' and eval_template == 'dapo' and step == 20"
    ).iloc[0]
    assert ttrl_dapo["macro_accuracy"] == 0.20
    assert ttrl_dapo["output_length_mean"] == 275.0
    assert ttrl_dapo["truncation_rate"] == 0.275
    step_zero = exported.query("step == 0")
    assert len(step_zero) == 4
    assert set(step_zero.query("eval_template == 'ttrl'")["macro_accuracy"]) == {0.04}
    assert set(step_zero.query("eval_template == 'dapo'")["macro_accuracy"]) == {0.03}
    assert set(step_zero.query("eval_template == 'ttrl'")["output_length_mean"]) == {400.0}
    assert set(step_zero.query("eval_template == 'dapo'")["output_length_mean"]) == {300.0}
