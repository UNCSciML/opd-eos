import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_student_teacher_eos_train_curves.py"


def test_cli_combines_student_and_teacher_eos_as_linear_one_by_four(tmp_path: Path) -> None:
    student_csv = tmp_path / "student.csv"
    teacher_csv = tmp_path / "teacher.csv"
    rows = [
        {"train_template": "ttrl", "step": 1},
        {"train_template": "ttrl", "step": 2},
        {"train_template": "dapo", "step": 1},
        {"train_template": "dapo", "step": 2},
    ]
    pd.DataFrame(
        [
            {
                **row,
                "student_prob_endoftext_last_token": endoftext,
                "student_prob_im_end_last_token": im_end,
            }
            for row, endoftext, im_end in zip(
                rows,
                (0.2, 0.4, 0.6, 0.8),
                (0.1, 0.2, 0.3, 0.4),
            )
        ]
    ).to_csv(student_csv, index=False)
    pd.DataFrame(
        [
            {
                **row,
                "teacher_prob_endoftext_last_token": endoftext,
                "teacher_prob_im_end_last_token": im_end,
            }
            for row, endoftext, im_end in zip(
                rows,
                (0.15, 0.3, 0.45, 0.6),
                (0.05, 0.1, 0.15, 0.2),
            )
        ]
    ).to_csv(teacher_csv, index=False)
    output_prefix = tmp_path / "figure" / "student-teacher-eos"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--student-csv",
            str(student_csv),
            "--teacher-csv",
            str(teacher_csv),
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
        assert image.width >= 5000
        assert image.width / image.height > 3.5

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        assert figure_text.count("TTRL Template") == 2
        assert figure_text.count("DAPO Template") == 2
        assert figure_text.count("<|endoftext|>") == 1
        assert figure_text.count("<|im_end|>") == 1
        assert "Teacher P(" not in figure_text
        assert "Student P(" not in figure_text
        assert "10−" not in figure_text
        assert "(a)" not in figure_text
