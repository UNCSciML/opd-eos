import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "analysis" / "plot_teacher_eos_diagnostic.py"


def _write_step(path: Path, step: int, p_e1, p_e2, response_length) -> None:
    p_e1 = np.asarray(p_e1, dtype=np.float64)
    p_e2 = np.asarray(p_e2, dtype=np.float64)
    eos_mass = p_e1 + p_e2
    np.savez(
        path / f"step_{step:04d}_teacher_eos_at_student_eos.npz",
        global_step=np.asarray(step),
        eos_token_ids=np.asarray([151643, 151645]),
        response_length=np.asarray(response_length),
        teacher_prob_e1=p_e1,
        teacher_prob_e2=p_e2,
        teacher_eos_mass=eos_mass,
        teacher_e1_share=p_e1 / eos_mass,
        teacher_e2_share=p_e2 / eos_mass,
        logprob_e2_minus_e1=np.log(p_e2) - np.log(p_e1),
    )


def test_cli_combines_all_steps_and_reports_hand_checked_statistics(tmp_path: Path) -> None:
    """Catch dropping a step or swapping the two EOS-token probability columns."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_step(input_dir, 1, [0.01, 0.10], [0.09, 0.10], [7, 11])
    _write_step(input_dir, 2, [0.20], [0.80], [13])
    output_prefix = tmp_path / "figure" / "teacher_eos_mismatch"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-prefix",
            str(output_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = pd.read_csv(output_prefix.with_suffix(".csv"))
    assert exported[["global_step", "sample_index"]].values.tolist() == [
        [1, 0],
        [1, 1],
        [2, 0],
    ]
    assert exported["teacher_prob_e1"].tolist() == [0.01, 0.10, 0.20]
    assert exported["teacher_prob_e2"].tolist() == [0.09, 0.10, 0.80]

    summary = json.loads(output_prefix.with_suffix(".json").read_text())
    assert summary["count"] == 3
    assert summary["e2_probability_greater_fraction"] == 2 / 3
    assert np.isclose(summary["teacher_e2_share_mean"], 2.2 / 3)
    assert np.isclose(summary["teacher_e2_share_median"], 0.8)
    assert np.isclose(summary["logprob_e2_minus_e1_mean"], np.log(36) / 3)

    assert output_prefix.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
    assert output_prefix.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with Image.open(output_prefix.with_suffix(".png")) as image:
        assert image.width >= 2800
        assert 1.3 < image.width / image.height < 1.8

    if pdftotext := shutil.which("pdftotext"):
        figure_text = subprocess.run(
            [pdftotext, str(output_prefix.with_suffix(".pdf")), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert figure_text.count("<|endoftext|>") == 1
        assert figure_text.count("<|im_end|>") == 1
        assert "151643" not in figure_text
        assert "151645" not in figure_text
        assert "Teacher EOS distributions" not in figure_text
        assert "Conditioned on" not in figure_text
        assert "median" not in figure_text
