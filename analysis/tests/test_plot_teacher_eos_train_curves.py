import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_teacher_eos_train_curves.py"


def _write_log(path: Path, rows: list[tuple[int, float, float]]) -> None:
    path.write_text(
        "\n".join(
            " - ".join(
                [
                    f"step:{step}",
                    f"teacher_eos_prob/endoftext_151643/last_token:{endoftext}",
                    f"teacher_eos_prob/im_end_151645/last_token:{im_end}",
                    f"training/global_step:{step}",
                ]
            )
            for step, endoftext, im_end in rows
        )
        + "\n"
    )


def test_cli_exports_both_teacher_eos_curves_without_token_ids_in_figure(tmp_path: Path) -> None:
    ttrl_log = tmp_path / "ttrl.out"
    dapo_log = tmp_path / "dapo.out"
    _write_log(ttrl_log, [(1, 1e-7, 0.25), (2, 1e-8, 0.10)])
    _write_log(dapo_log, [(1, 2e-7, 0.35), (2, 2e-9, 1e-4)])
    output_prefix = tmp_path / "figure" / "teacher-eos-curves"

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
        "teacher_prob_endoftext_last_token",
        "teacher_prob_im_end_last_token",
    ]
    assert len(exported) == 4
    row = exported.query("train_template == 'dapo' and step == 2").iloc[0]
    assert row["teacher_prob_endoftext_last_token"] == 2e-9
    assert row["teacher_prob_im_end_last_token"] == 1e-4

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width >= 3800
        assert 2.2 < image.width / image.height < 3.2

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        assert figure_text.count("<|endoftext|>") == 1
        assert figure_text.count("<|im_end|>") == 1
        assert "151643" not in figure_text
        assert "151645" not in figure_text


def test_cli_linear_scale_uses_decimal_probability_axis(tmp_path: Path) -> None:
    ttrl_log = tmp_path / "ttrl.out"
    dapo_log = tmp_path / "dapo.out"
    _write_log(ttrl_log, [(1, 0.01, 0.25), (2, 0.02, 0.10)])
    _write_log(dapo_log, [(1, 0.03, 0.35), (2, 0.04, 0.20)])
    output_prefix = tmp_path / "figure" / "teacher-eos-curves-linear"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            f"ttrl,{ttrl_log}",
            "--run",
            f"dapo,{dapo_log}",
            "--yscale",
            "linear",
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(output_prefix.with_suffix(".pdf")), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "10−" not in figure_text
        assert "0.0" in figure_text
