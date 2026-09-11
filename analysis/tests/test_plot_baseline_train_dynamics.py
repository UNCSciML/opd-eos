import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_baseline_train_dynamics.py"


def _write_log(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    path.write_text(
        "\n".join(
            " - ".join(
                [
                    f"step:{step}",
                    f"response_length/mean:{mean}",
                    f"response_length/clip_ratio:{clip}",
                    f"student_eos_prob/endoftext_151643/last_token:{endoftext}",
                    f"student_eos_prob/im_end_151645/last_token:{im_end}",
                    f"training/global_step:{step}",
                ]
            )
            for step, mean, clip, endoftext, im_end in rows
        )
        + "\n"
    )


def test_cli_exports_raw_steps_and_splits_training_and_eos_figures(tmp_path: Path) -> None:
    ttrl_log = tmp_path / "ttrl.out"
    dapo_log = tmp_path / "dapo.out"
    _write_log(ttrl_log, [(1, 100.0, 0.25, 0.5, 1e-6), (2, 200.0, 0.50, 0.1, 1e-8)])
    _write_log(dapo_log, [(1, 300.0, 0.75, 0.4, 2e-6), (2, 400.0, 1.00, 0.01, 2e-9)])

    output_prefix = tmp_path / "figure" / "train-dynamics"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            f"ttrl,{ttrl_log}",
            "--run",
            f"dapo,{dapo_log}",
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    assert list(exported.columns) == [
        "train_template",
        "step",
        "response_length_mean",
        "response_length_clip_ratio",
        "student_prob_endoftext_last_token",
        "student_prob_im_end_last_token",
    ]
    assert len(exported) == 4
    row = exported.query("train_template == 'ttrl' and step == 2").iloc[0]
    assert row["response_length_mean"] == 200.0
    assert row["response_length_clip_ratio"] == 0.5
    assert row["student_prob_endoftext_last_token"] == 0.1
    assert row["student_prob_im_end_last_token"] == 1e-8

    training_png = output_prefix.parent / f"{output_prefix.name}_1x4.png"
    training_pdf = output_prefix.parent / f"{output_prefix.name}_1x4.pdf"
    eos_png = output_prefix.parent / f"{output_prefix.name}_eos_1x2.png"
    eos_pdf = output_prefix.parent / f"{output_prefix.name}_eos_1x2.pdf"
    for png in (training_png, eos_png):
        assert png.read_bytes().startswith(b"\x89PNG")
    for pdf in (training_pdf, eos_pdf):
        assert pdf.read_bytes().startswith(b"%PDF")

    with Image.open(training_png) as image:
        assert image.width >= 5000
        assert image.width / image.height > 3.0
    with Image.open(eos_png) as image:
        assert image.width >= 2500
        assert 1.5 < image.width / image.height < 3.0

    if pdftotext := shutil.which("pdftotext"):
        training_text = subprocess.run(
            [pdftotext, str(training_pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        eos_text = subprocess.run(
            [pdftotext, str(eos_pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        assert "Mean response length" in training_text
        assert "Clipped responses" in training_text
        assert "TTRL Template" in training_text
        assert "DAPO Template" in training_text
        assert "TTRL Train" not in training_text
        assert "DAPO Train" not in training_text
        assert "<|endoftext|>" not in training_text
        assert "<|im_end|>" not in training_text
        assert "TTRL Template" in eos_text
        assert "DAPO Template" in eos_text
        assert "TTRL Train" not in eos_text
        assert "DAPO Train" not in eos_text
        assert "<|endoftext|>" in eos_text
        assert "<|im_end|>" in eos_text
        assert "151643" not in eos_text
        assert "151645" not in eos_text
