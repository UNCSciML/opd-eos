import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_semantic_vs_baseline_template_matrix.py"


def _write_fixture_inputs(baseline_csv: Path, semantic_csv: Path) -> None:
    baseline_rows = []
    semantic_rows = []
    for eval_template in ("ttrl", "dapo"):
        for train_template in ("ttrl", "dapo"):
            for step in (0, 200):
                baseline_rows.append(
                    {
                        "train_template": train_template,
                        "eval_template": eval_template,
                        "step": step,
                        "macro_accuracy": 0.10 + step / 10000,
                        "output_length_mean": 1000.0 + step,
                    }
                )
        for train_template in ("ttrl", "dapo", "eopd"):
            for step in (0, 200):
                semantic_rows.append(
                    {
                        "train_template": train_template,
                        "eval_template": eval_template,
                        "step": step,
                        "macro_mean_score": 0.12 + step / 10000,
                        "macro_avg_output_length": 9999.0,
                        "AIME24_avg_output_length": 100.0 + step,
                        "AIME25_avg_output_length": 200.0 + step,
                        "AMC23_avg_output_length": 400.0 + step,
                    }
                )
    pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
    pd.DataFrame(semantic_rows).to_csv(semantic_csv, index=False)


def test_cli_compares_only_shared_templates_with_consistent_length_aggregation(
    tmp_path: Path,
) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    semantic_csv = tmp_path / "semantic.csv"
    _write_fixture_inputs(baseline_csv, semantic_csv)
    output_prefix = tmp_path / "semantic-vs-no-fix"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--semantic-csv",
            str(semantic_csv),
            "--baseline-csv",
            str(baseline_csv),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    assert len(exported) == 16
    assert set(exported["method"]) == {"semantic_fix", "no_eos_fix"}
    assert set(exported["train_template"]) == {"ttrl", "dapo"}
    assert set(exported["eval_template"]) == {"ttrl", "dapo"}
    assert set(exported["step"]) == {0, 200}

    semantic_step_zero = exported.query(
        "method == 'semantic_fix' and train_template == 'ttrl' "
        "and eval_template == 'ttrl' and step == 0"
    ).iloc[0]
    assert semantic_step_zero["macro_accuracy"] == 0.12
    assert semantic_step_zero["output_length_mean"] == 250.0
    baseline_step_zero = exported.query(
        "method == 'no_eos_fix' and train_template == 'ttrl' "
        "and eval_template == 'ttrl' and step == 0"
    ).iloc[0]
    assert baseline_step_zero["output_length_mean"] == 1000.0

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width / image.height > 4.0

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        for title in (
            "Semantic fix: Accuracy",
            "Semantic fix: Length",
            "No EOS fix: Accuracy",
            "No EOS fix: Length",
        ):
            assert title in figure_text
        for label in ("TTRL", "DAPO"):
            assert label in figure_text
        assert "EOPD" not in figure_text
