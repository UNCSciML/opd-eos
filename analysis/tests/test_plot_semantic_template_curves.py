import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from analysis import plot_semantic_template_curves as plotter


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_semantic_template_curves.py"


def test_build_figure_can_disable_late_checkpoint_shading() -> None:
    records = pd.DataFrame(
        [
            {
                "train_template": train_template,
                "eval_template": eval_template,
                "step": 20,
                "macro_mean_score": 0.1,
                "macro_avg_output_length": 1000.0,
            }
            for eval_template in ("ttrl", "dapo")
            for train_template in ("ttrl", "dapo", "eopd")
        ]
    )

    figure = plotter.build_figure(records, shade_late_window=False)
    try:
        assert all(len(axis.patches) == 0 for axis in figure.axes)
    finally:
        plt.close(figure)


def test_cli_plots_score_and_mean_length_for_both_eval_templates_as_one_by_four(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "semantic-template-curves.csv"
    rows = []
    for eval_index, eval_template in enumerate(("ttrl", "dapo"), start=1):
        for train_index, train_template in enumerate(("ttrl", "dapo", "eopd"), start=1):
            for step in (20, 200):
                rows.append(
                    {
                        "train_template": train_template,
                        "eval_template": eval_template,
                        "step": step,
                        "macro_mean_score": 0.10 + 0.01 * train_index + 0.001 * eval_index,
                        "macro_avg_output_length": 1000 * train_index + step + 10 * eval_index,
                    }
                )
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    output_prefix = tmp_path / "figures" / "semantic-template-score-curves"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-csv",
            str(input_csv),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width >= 4000
        assert image.width / image.height > 4.9

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        assert figure_text.count("Fixed TTRL eval template") == 2
        assert figure_text.count("Fixed DAPO eval template") == 2
        assert figure_text.count("Macro score") == 2
        assert figure_text.count("Mean response length (tokens)") == 2
        for label in ("TTRL train", "DAPO train", "Raw-question train"):
            assert figure_text.count(label) == 1
        assert "EOPD train" not in figure_text
        assert "Semantic-class: train-template comparison (@16)" not in figure_text


def test_cli_replaces_only_dapo_scores_with_official_results(tmp_path: Path) -> None:
    """Catch changing TTRL scores or response lengths while replacing DAPO scores."""
    input_csv = tmp_path / "semantic-template-curves.csv"
    rows = []
    for eval_template in ("ttrl", "dapo"):
        for train_index, train_template in enumerate(("ttrl", "dapo", "eopd"), start=1):
            rows.append(
                {
                    "train_template": train_template,
                    "eval_template": eval_template,
                    "step": 20,
                    "macro_mean_score": 0.10 + 0.01 * train_index,
                    "macro_avg_output_length": 1000.0 * train_index,
                }
            )
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    official_csv = tmp_path / "official-dapo.csv"
    pd.DataFrame(
        [
            {"train_template": "ttrl", "step": 20, "official_macro_accuracy": 0.01},
            {"train_template": "dapo", "step": 20, "official_macro_accuracy": 0.02},
            {"train_template": "eopd", "step": 20, "official_macro_accuracy": 0.03},
        ]
    ).to_csv(official_csv, index=False)

    output_prefix = tmp_path / "figures" / "semantic-template-official-dapo"
    step_zero_csv = tmp_path / "step-zero.csv"
    pd.DataFrame(
        [
            {
                "eval_template": "ttrl",
                "step": 0,
                "macro_accuracy": 0.04,
                "output_length_mean": 400.0,
                "macro_avg_output_length": 410.0,
                "truncation_rate": 0.40,
            },
            {
                "eval_template": "dapo",
                "step": 0,
                "macro_accuracy": 0.03,
                "output_length_mean": 300.0,
                "macro_avg_output_length": 310.0,
                "truncation_rate": 0.30,
            },
        ]
    ).to_csv(step_zero_csv, index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-csv",
            str(input_csv),
            "--official-dapo-csv",
            str(official_csv),
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
    plotted = pd.read_csv(output_prefix.with_suffix(".csv"))
    ttrl_eval = plotted.query(
        "train_template == 'ttrl' and eval_template == 'ttrl' and step == 20"
    ).iloc[0]
    ttrl_dapo = plotted.query(
        "train_template == 'ttrl' and eval_template == 'dapo' and step == 20"
    ).iloc[0]
    dapo_dapo = plotted.query(
        "train_template == 'dapo' and eval_template == 'dapo' and step == 20"
    ).iloc[0]
    eopd_dapo = plotted.query(
        "train_template == 'eopd' and eval_template == 'dapo' and step == 20"
    ).iloc[0]
    assert ttrl_eval["macro_mean_score"] == 0.11
    assert ttrl_dapo["macro_mean_score"] == 0.01
    assert dapo_dapo["macro_mean_score"] == 0.02
    assert eopd_dapo["macro_mean_score"] == 0.03
    assert ttrl_dapo["macro_avg_output_length"] == 1000.0
    step_zero = plotted.query("step == 0")
    assert len(step_zero) == 6
    assert set(step_zero.query("eval_template == 'ttrl'")["macro_mean_score"]) == {0.04}
    assert set(step_zero.query("eval_template == 'dapo'")["macro_mean_score"]) == {0.03}
    assert set(step_zero.query("eval_template == 'ttrl'")["macro_avg_output_length"]) == {410.0}
    assert set(step_zero.query("eval_template == 'dapo'")["macro_avg_output_length"]) == {310.0}
