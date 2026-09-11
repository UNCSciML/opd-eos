import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"


def run_dry_launcher(**overrides):
    environment = {
        **os.environ,
        "DRY_RUN": "true",
        "SLURM_JOB_ID": "unit-test",
        **{key: str(value) for key, value in overrides.items()},
    }
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launcher_selects_eopd_dataset_and_canonical_eos_without_changing_defaults():
    result = run_dry_launcher(TRAIN_TEMPLATE="eopd", EOS_MODE="canonical")

    assert result.returncode == 0, result.stderr
    assert "train_template=eopd" in result.stdout
    assert "dapo-math-17k-raw-question.parquet" in result.stdout
    assert "reward_score/opd_math/__init__.py" in result.stdout
    assert "eos_mode=canonical" in result.stdout
    assert "additional_eos_token_ids=" in result.stdout
    assert "train_batch_size=16" in result.stdout
    assert "enable_thinking=False" in result.stdout
    assert "log_prob_top_k=0" in result.stdout
    assert f"project_root={PROJECT_ROOT}" in result.stdout
    assert "actor_model_path=Qwen/Qwen3-1.7B-Base" in result.stdout
    assert "reward_model_path=Qwen/Qwen3-4B" in result.stdout
    assert "wandb_project=OPD Length Inflation" in result.stdout


def test_launcher_derives_terminal_and_additional_ids_from_the_manifest():
    for mode in ("two_stop", "semantic_class"):
        result = run_dry_launcher(EOS_MODE=mode)
        assert result.returncode == 0, result.stderr
        assert "semantic_eos_token_ids=auto" in result.stdout
        assert "additional_eos_token_ids=manifest-derived" in result.stdout


def test_launcher_accepts_model_and_template_sources_without_hardcoded_eos_ids():
    result = run_dry_launcher(
        ACTOR_MODEL_PATH="org/base-model",
        REWARD_MODEL_PATH="org/instruct-model",
        CHAT_TEMPLATE_MODEL="org/instruct-model",
        EOS_MODE="semantic_class",
    )

    assert result.returncode == 0, result.stderr
    assert "actor_model_path=org/base-model" in result.stdout
    assert "reward_model_path=org/instruct-model" in result.stdout
    assert "chat_template_model=org/instruct-model" in result.stdout
    assert "semantic_eos_token_ids=auto" in result.stdout


def test_launcher_rejects_unknown_template_or_eos_mode():
    assert run_dry_launcher(TRAIN_TEMPLATE="unknown").returncode != 0
    assert run_dry_launcher(EOS_MODE="unknown").returncode != 0
