import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_baseline_official_dapo_vs_teacher.py"


def _write_teacher_task(path: Path, responses: list[str]) -> None:
    rows = []
    for example_id, response in enumerate(responses):
        rows.append(
            {
                "example_id": example_id,
                "answer": "\\boxed{4}",
                "response": response,
                "seed": 0,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_plots_original_and_adapted_grader_panels_without_titles(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "train_template": train_template,
                "eval_template": eval_template,
                "step": step,
                "macro_accuracy": accuracy,
                "output_length_mean": 1000.0,
                "truncation_rate": 0.0,
            }
            for train_template, eval_template, step, accuracy in [
                ("ttrl", "dapo", 0, 0.10),
                ("ttrl", "dapo", 200, 0.20),
                ("dapo", "dapo", 0, 0.10),
                ("dapo", "dapo", 200, 0.30),
                # These non-DAPO-eval rows must not enter the exported comparison.
                ("ttrl", "ttrl", 0, 0.90),
                ("dapo", "ttrl", 0, 0.80),
            ]
        ]
    ).to_csv(baseline_csv, index=False)
    adapted_baseline_csv = tmp_path / "adapted-baseline.csv"
    pd.DataFrame(
        [
            {
                "train_template": train_template,
                "eval_template": "dapo",
                "step": step,
                "macro_accuracy": accuracy,
            }
            for train_template, step, accuracy in [
                ("ttrl", 0, 0.40),
                ("ttrl", 200, 0.50),
                ("dapo", 0, 0.40),
                ("dapo", 200, 0.60),
            ]
        ]
    ).to_csv(adapted_baseline_csv, index=False)

    teacher_root = tmp_path / "teacher"
    teacher_step = teacher_root / "step_0000"
    teacher_step.mkdir(parents=True)
    (teacher_root / "eval_run_config.json").write_text(
        json.dumps({"prompt_template": "dapo"}), encoding="utf-8"
    )
    _write_teacher_task(
        teacher_step / "aime24_test.jsonl",
        ["Reasoning.\nAnswer: 4", "Reasoning.\nAnswer: 5"],
    )
    _write_teacher_task(
        teacher_step / "aime25_test.jsonl",
        ["Reasoning.\nAnswer: $$4$$", "Reasoning.\nAnswer: $$\\boxed{4}$$"],
    )
    _write_teacher_task(
        teacher_step / "amc23_test.jsonl",
        ["Reasoning.\nAnswer: 4", "Reasoning.\nAnswer: \\boxed{4}"],
    )

    output_prefix = tmp_path / "figure" / "official-dapo-vs-teacher"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-csv",
            str(baseline_csv),
            "--adapted-baseline-csv",
            str(adapted_baseline_csv),
            "--teacher-eval-root",
            str(teacher_root),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    original_baseline = exported.query("series_type == 'baseline' and grader == 'original_official_dapo'")
    assert len(original_baseline) == 4
    assert set(original_baseline["train_template"]) == {"ttrl", "dapo"}
    assert set(original_baseline["eval_template"]) == {"dapo"}
    assert set(original_baseline["step"]) == {0, 200}
    assert set(original_baseline["macro_accuracy"]) == {0.10, 0.20, 0.30}

    adapted_baseline = exported.query("series_type == 'baseline' and grader == 'answer_robust_dapo'")
    assert len(adapted_baseline) == 4
    assert set(adapted_baseline["macro_accuracy"]) == {0.40, 0.50, 0.60}

    original_teacher = exported.query(
        "series_type == 'teacher_reference' and grader == 'original_official_dapo'"
    )
    assert len(original_teacher) == 2
    assert set(original_teacher["step"]) == {0, 200}
    # Original per-task scores are 0.5, 0.0, and 1.0.
    assert set(original_teacher["macro_accuracy"]) == {0.5}
    adapted_teacher = exported.query(
        "series_type == 'teacher_reference' and grader == 'answer_robust_dapo'"
    )
    assert len(adapted_teacher) == 2
    # The adapted grader additionally accepts both immediate $$...$$ answer blocks.
    assert set(adapted_teacher["macro_accuracy"]) == {5 / 6}

    assert output_prefix.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert output_prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with Image.open(output_prefix.with_suffix(".png")) as image:
        assert image.width >= 5000
        assert 2.5 < image.width / image.height < 3.4

    pdf_text = subprocess.run(
        ["pdftotext", str(output_prefix.with_suffix(".pdf")), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "TTRL-trained baseline" in pdf_text
    assert "DAPO-trained baseline" in pdf_text
    assert "Qwen3-4B teacher" in pdf_text
    assert "50.00%" in pdf_text
    assert "83.33%" in pdf_text
    assert "Original DAPO Avg@16 (%)" in pdf_text
    assert "Adapted DAPO Avg@16 (%)" in pdf_text
    assert "Baseline OPD under the original DAPO grader" not in pdf_text


def test_cli_rejects_non_dapo_teacher_prompt(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "train_template": train,
                "eval_template": "dapo",
                "step": step,
                "macro_accuracy": 0.1,
            }
            for train in ("ttrl", "dapo")
            for step in (0, 200)
        ]
    ).to_csv(baseline_csv, index=False)
    adapted_baseline_csv = tmp_path / "adapted-baseline.csv"
    pd.read_csv(baseline_csv).to_csv(adapted_baseline_csv, index=False)
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    (teacher_root / "eval_run_config.json").write_text(
        json.dumps({"prompt_template": "ttrl"}), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline-csv",
            str(baseline_csv),
            "--adapted-baseline-csv",
            str(adapted_baseline_csv),
            "--teacher-eval-root",
            str(teacher_root),
            "--output-prefix",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Expected a DAPO-prompt teacher eval root" in result.stderr
