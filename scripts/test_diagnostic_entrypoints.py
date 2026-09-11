import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EOS_SCORING_CHECK = PROJECT_ROOT / "slurm" / "diagnostic" / "verify_teacher_eos_scoring_1gpu.sl"


def run_script(path: Path, **overrides) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "DRY_RUN": "true",
        "SLURM_JOB_ID": "unit-test",
        **{key: str(value) for key, value in overrides.items()},
    }
    return subprocess.run(
        ["bash", str(path)], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize(
    ("pair", "student", "teacher"),
    [
        ("gemma", "google/gemma-3-4b-pt", "google/gemma-3-4b-it"),
        ("qwen", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B"),
        ("llama", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ],
)
def test_eos_scoring_check_selects_each_pair(pair, student, teacher):
    result = run_script(EOS_SCORING_CHECK, PAIR=pair)

    assert result.returncode == 0, result.stderr
    assert f"pair={pair}" in result.stdout
    assert f"student={student}" in result.stdout
    assert f"teacher={teacher}" in result.stdout


def test_eos_scoring_check_rejects_an_unknown_pair():
    result = run_script(EOS_SCORING_CHECK, PAIR="mistral")

    assert result.returncode == 2
    assert "gemma, qwen or llama" in result.stderr


def test_eos_scoring_check_is_read_only_and_needs_one_gpu():
    script = EOS_SCORING_CHECK.read_text()
    prose = " ".join(script.split())

    assert "--gres=gpu:1" in script
    assert "Read-only: trains # nothing and writes no checkpoints" in prose
    assert "/work/users/" not in script
    assert "SLURM_SUBMIT_DIR" in script
