import os
import shutil
import subprocess
import sys
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "slurm" / "train"
EVAL_MODEL = PROJECT_ROOT / "slurm" / "eval" / "eval_model_4x6000pro.sl"
SMOKE_EVAL = PROJECT_ROOT / "slurm" / "smoke" / "smoke_eval_base_4x6000pro.sl"
CHECKPOINT_EVAL = PROJECT_ROOT / "eval_opd_ttrl_qwen3_1p7b_ckpts20_200_batched_n16_4gpu.sl"
SWEEP_SUBMIT = PROJECT_ROOT / "slurm" / "submit_ttrl_eos_sweep.sh"
EOS_DIAGNOSTIC = PROJECT_ROOT / "slurm" / "diagnostic" / "diagnose_baseline_teacher_eos_5step_4x6000pro.sl"
BASE_TEMPLATE_EVAL = PROJECT_ROOT / "slurm" / "eval" / "eval_base_step0_templates_2x6000pro.sl"


def run_script(path: Path, **overrides) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "DRY_RUN": "true",
        "SLURM_JOB_ID": "unit-test",
        **{key: str(value) for key, value in overrides.items()},
    }
    return subprocess.run(
        ["bash", str(path)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("filename", "template", "mode"),
    [
        ("train_ttrl_eos_a_baseline_200step.sl", "ttrl", "baseline"),
        ("train_ttrl_eos_b_two_stop_200step.sl", "ttrl", "two_stop"),
        ("train_ttrl_eos_c_teacher_map_200step.sl", "ttrl", "teacher_map"),
        ("train_ttrl_eos_d_semantic_class_200step.sl", "ttrl", "semantic_class"),
        ("train_ttrl_eos_e_canonical_200step.sl", "ttrl", "canonical"),
        ("train_dapo_baseline_200step.sl", "dapo", "baseline"),
        ("train_eopd_baseline_200step.sl", "eopd", "baseline"),
    ],
)
def test_each_formal_train_entrypoint_selects_one_condition_and_keeps_200_step_defaults(
    filename, template, mode
):
    result = run_script(TRAIN_DIR / filename)

    assert result.returncode == 0, result.stderr
    assert f"train_template={template}" in result.stdout
    assert f"eos_mode={mode}" in result.stdout
    assert "total_training_steps=200" in result.stdout
    assert "train_batch_size=16" in result.stdout
    assert "rollout_n=4" in result.stdout
    assert "save_freq=20" in result.stdout
    assert "enable_thinking=False" in result.stdout
    assert "log_prob_top_k=0" in result.stdout


@pytest.mark.parametrize(
    ("mode", "expected_stops", "expected_blocked"),
    [
        ("baseline", "model-default", ""),
        ("two_stop", "151643,151645", ""),
        ("teacher_map", "151643", ""),
        ("semantic_class", "151643,151645", ""),
        ("canonical", "151643", "151645"),
    ],
)
def test_direct_model_eval_resolves_generation_semantics_from_explicit_settings(
    tmp_path, mode, expected_stops, expected_blocked
):
    model_path = tmp_path / "hf_model"
    model_path.mkdir()

    result = run_script(
        EVAL_MODEL,
        MODEL_PATH=model_path,
        EOS_MODE=mode,
        EVAL_TEMPLATE="dapo",
    )

    assert result.returncode == 0, result.stderr
    assert f"model_path={model_path}" in result.stdout
    assert f"eos_mode={mode}" in result.stdout
    assert f"stop_token_ids={expected_stops}" in result.stdout
    assert f"blocked_token_ids={expected_blocked}" in result.stdout
    assert "enable_thinking=false" in result.stdout
    assert "eval_template=dapo" in result.stdout
    assert "tasks=AIME24,AIME25,AMC23" in result.stdout
    assert "n=16" in result.stdout
    assert "max_tokens=8192" in result.stdout
    assert "batched_n_sampling=true" in result.stdout


def test_base_eval_smoke_is_only_a_small_override_of_the_formal_model_eval():
    result = run_script(SMOKE_EVAL)

    assert result.returncode == 0, result.stderr
    assert "model_path=Qwen/Qwen3-1.7B-Base" in result.stdout
    assert "eos_mode=baseline" in result.stdout
    assert "stop_token_ids=model-default" in result.stdout
    assert "tasks=AMC23" in result.stdout
    assert "n=1" in result.stdout
    assert "max_tokens=256" in result.stdout
    assert "batched_n_sampling=true" in result.stdout


@pytest.mark.parametrize(("case_id", "template"), [(0, "ttrl"), (1, "dapo")])
def test_base_step0_template_eval_uses_one_default_eos_and_formal_settings(case_id, template):
    result = run_script(
        BASE_TEMPLATE_EVAL,
        SLURM_ARRAY_TASK_ID=case_id,
        CONDA_ENV=Path(sys.executable).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert "model_path=Qwen/Qwen3-1.7B-Base" in result.stdout
    assert "eos_mode=baseline" in result.stdout
    assert "stop_token_ids=model-default" in result.stdout
    assert "blocked_token_ids=" in result.stdout
    assert "enable_thinking=false" in result.stdout
    assert f"eval_template={template}" in result.stdout
    assert "tasks=AIME24,AIME25,AMC23" in result.stdout
    assert "n=16" in result.stdout
    assert "max_tokens=8192" in result.stdout
    assert "max_model_len=12288" in result.stdout
    assert "temperature=0.7" in result.stdout
    assert "top_p=0.95" in result.stdout
    assert "batched_n_sampling=true" in result.stdout


def test_teacher_eos_diagnostic_is_baseline_ttrl_without_updates_or_checkpoints():
    result = run_script(EOS_DIAGNOSTIC)

    assert result.returncode == 0, result.stderr
    assert "train_template=ttrl" in result.stdout
    assert "eos_mode=baseline" in result.stdout
    assert "train_batch_size=16" in result.stdout
    assert "rollout_n=4" in result.stdout
    assert "total_training_steps=5" in result.stdout
    assert "save_freq=-1" in result.stdout
    assert "test_freq=-1" in result.stdout
    assert "critic_warmup=6" in result.stdout
    assert "wandb_mode=disabled" in result.stdout
    assert "eos_diagnostic_output_dir=" in result.stdout


def test_train_entrypoint_discovers_project_root_after_repository_is_moved(tmp_path):
    moved_root = tmp_path / "renamed-repository"
    moved_train_dir = moved_root / "slurm" / "train"
    moved_train_dir.mkdir(parents=True)
    shutil.copy2(
        TRAIN_DIR / "train_ttrl_eos_a_baseline_200step.sl",
        moved_train_dir / "train_ttrl_eos_a_baseline_200step.sl",
    )
    shutil.copy2(
        PROJECT_ROOT / "run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl",
        moved_root / "run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl",
    )

    result = run_script(moved_train_dir / "train_ttrl_eos_a_baseline_200step.sl")

    assert result.returncode == 0, result.stderr
    assert f"project_root={moved_root}" in result.stdout


def test_checkpoint_eval_dry_run_inherits_all_generation_settings_from_train_manifest(tmp_path):
    checkpoint_root = tmp_path / "checkpoint" / "example-run"
    checkpoint_root.mkdir(parents=True)
    model_path = tmp_path / "student-model"
    model_path.mkdir()
    semantics = {
        "schema_version": 2,
        "model_path": str(model_path),
        "enable_thinking": False,
        "prompt_template": "eopd",
        "eos_mode": "canonical",
        "semantic_eos_token_ids": [151643, 151645],
        "rollout_stop_token_ids": [151643],
        "blocked_token_ids": [151645],
    }
    (checkpoint_root / "run_semantics.json").write_text(json.dumps(semantics))

    result = run_script(
        CHECKPOINT_EVAL,
        CHECKPOINT_ROOT=checkpoint_root,
        CONDA_ENV=Path(sys.executable).resolve().parent.parent,
        STEPS="20",
    )

    assert result.returncode == 0, result.stderr
    assert f"project_root={PROJECT_ROOT}" in result.stdout
    assert f"base_model={model_path}" in result.stdout
    assert "eos_mode=canonical" in result.stdout
    assert "stop_token_ids=151643" in result.stdout
    assert "blocked_token_ids=151645" in result.stdout
    assert "enable_thinking=false" in result.stdout
    assert "eval_template=eopd" in result.stdout
    assert "tasks=AIME24,AIME25,AMC23" in result.stdout
    assert "n=16" in result.stdout
    assert "save_token_ids=true" in result.stdout
    assert "wandb_project=OPD Length Inflation" in result.stdout


def test_ttrl_eos_sweep_dry_run_chains_one_matching_eval_after_each_train():
    result = run_script(SWEEP_SUBMIT, SWEEP_TAG="test-sweep")

    assert result.returncode == 0, result.stderr
    for mode in ("baseline", "two_stop", "teacher_map", "semantic_class", "canonical"):
        experiment = f"opd_ttrl_{mode}_sampled_q17b_q4b_bs16_n4_200step_test-sweep"
        assert f"mode={mode} train_job=DRY_RUN" in result.stdout
        assert f"checkpoint/{experiment}" in result.stdout
        assert f"dependency=afterok:DRY_RUN-{mode}" in result.stdout
    assert result.stdout.count("eval_job=DRY_RUN") == 5


def test_submitted_wrapper_uses_slurm_submit_directory_instead_of_spool_location(tmp_path):
    spool_dir = tmp_path / "var" / "spool" / "slurm"
    spool_dir.mkdir(parents=True)
    spooled_script = spool_dir / "job_script"
    shutil.copy2(SMOKE_EVAL, spooled_script)
    model_path = tmp_path / "model"
    model_path.mkdir()
    environment = {
        **os.environ,
        "DRY_RUN": "true",
        "SLURM_JOB_ID": "unit-test",
        "SLURM_SUBMIT_DIR": str(PROJECT_ROOT),
        "MODEL_PATH": str(model_path),
    }

    result = subprocess.run(
        ["bash", str(spooled_script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"model_path={model_path}" in result.stdout
