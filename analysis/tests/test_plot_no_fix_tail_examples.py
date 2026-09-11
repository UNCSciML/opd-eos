import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_no_fix_tail_examples.py"


def _write_example(
    root: Path,
    step: int,
    dataset: str,
    example_id: int,
    seed: int,
    answer: str,
    response: str,
) -> None:
    directory = root / f"step_{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_t0.7_p0.95_n16-MNT8192"
    row = {
        "example_id": example_id,
        "prompt": "Synthetic math problem",
        "answer": answer,
        "seed": seed,
        "response": response,
    }
    (directory / f"{stem}.jsonl").write_text(json.dumps(row) + "\n")
    np.savez_compressed(
        directory / f"{stem}.tokens.npz",
        token_ids=np.zeros(8192, dtype=np.int32),
        offsets=np.asarray([0, 8192], dtype=np.int64),
    )


def test_cli_draws_three_real_output_patterns_and_exports_png_pdf(tmp_path: Path) -> None:
    eval_root = tmp_path / "eval"
    _write_example(
        eval_root,
        100,
        "amc23",
        10,
        8,
        "5",
        "The units digit is five.\n\\boxed{5}\n"
        "### Final Answer:\n\\boxed{5}\n### Final Answer:\n\\boxed{5}",
    )
    _write_example(
        eval_root,
        100,
        "aime24",
        7,
        1,
        r"\boxed{25}",
        "Therefore xy=25.\n\\boxed{25}\n"
        + "Let’s try $x = 2$, $y = 12.5$.\n" * 4,
    )
    _write_example(
        eval_root,
        200,
        "amc23",
        0,
        13,
        "27",
        "They meet 27 miles from City A.\n\\boxed{27}\n🚨 🚨 🚨 " + "✅ " * 8,
    )

    output_prefix = tmp_path / "figure" / "no-fix-tail-examples"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--eval-root",
            str(eval_root),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "missing from font" not in result.stderr
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width >= 3000
        assert 1.35 < image.width / image.height < 2.2

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(pdf), "-"], check=True, capture_output=True, text=True
        ).stdout
        for label in (
            "Repeated final answer",
            "Repeated reasoning",
            "Repeated emoji",
            "Expected stop",
            "8192-token cap",
            "U+1F6A8 ALERT",
            "U+2705 CHECK",
        ):
            assert label in figure_text
