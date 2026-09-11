import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "slurm" / "train"
LAUNCHER = PROJECT_ROOT / "run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"

MULTIMODEL_ENTRYPOINTS = [
    (
        "train_ttrl_llama32_3b_base_to_instruct_baseline_200step.sl",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "opd_ttrl_llama32_3b_base_to_instruct_baseline",
        "20000-24999",
        "True",
        "True",
        "baseline",
    ),
    (
        "train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl",
        "google/gemma-3-4b-pt",
        "google/gemma-3-4b-it",
        "opd_ttrl_gemma3_4bpt_to_4bit_baseline",
        "25000-29999",
        "False",
        "False",
        "baseline",
    ),
    (
        "train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "opd_ttrl_llama32_3b_base_to_instruct_semantic_class",
        "20000-24999",
        "True",
        "True",
        "semantic_class",
    ),
    (
        "train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl",
        "google/gemma-3-4b-pt",
        "google/gemma-3-4b-it",
        "opd_ttrl_gemma3_4bpt_to_4bit_semantic_class",
        "25000-29999",
        "False",
        "False",
        "semantic_class",
    ),
]

# The semantic terminal set must never be hard-coded per model: it is discovered
# from the student and teacher generation configs at submit time.
MULTIMODEL_HARD_CODED_EOS_IDS = ["151643", "151645", "128001", "128008", "128009", "106"]


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
    (
        "filename",
        "student",
        "teacher",
        "experiment_prefix",
        "ray_ports",
        "return_multi_modal_inputs",
        "enable_activation_offload",
        "eos_mode",
    ),
    MULTIMODEL_ENTRYPOINTS,
)
def test_multimodel_entrypoints_are_formal_ttrl_runs(
    filename,
    student,
    teacher,
    experiment_prefix,
    ray_ports,
    return_multi_modal_inputs,
    enable_activation_offload,
    eos_mode,
):
    result = run_script(TRAIN_DIR / filename)

    assert result.returncode == 0, result.stderr
    assert f"actor_model_path={student}" in result.stdout
    assert f"reward_model_path={teacher}" in result.stdout
    assert f"chat_template_model={teacher}" in result.stdout
    assert "train_template=ttrl" in result.stdout
    assert f"eos_mode={eos_mode}" in result.stdout
    assert "semantic_eos_token_ids=auto" in result.stdout
    assert "enable_thinking=False" in result.stdout
    assert "train_batch_size=16" in result.stdout
    assert "rollout_n=4" in result.stdout
    assert "max_response_length=7168" in result.stdout
    assert "ppo_max_token_len_per_gpu=8192" in result.stdout
    assert "reward_max_token_len_per_gpu=8192" in result.stdout
    assert "total_training_steps=200" in result.stdout
    assert "save_freq=20" in result.stdout
    assert "test_freq=-1" in result.stdout
    assert "wandb_mode=online" in result.stdout
    assert "wandb_project=OPD Length Inflation" in result.stdout
    assert f"ray_worker_port_range={ray_ports}" in result.stdout
    assert f"return_multi_modal_inputs={return_multi_modal_inputs}" in result.stdout
    assert f"enable_activation_offload={enable_activation_offload}" in result.stdout
    assert "log_prob_top_k=0" in result.stdout
    assert f"experiment_name={experiment_prefix}_unit-test" in result.stdout


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_ENTRYPOINTS]
)
def test_multimodel_entrypoints_are_portable_and_document_submission(filename):
    script = (TRAIN_DIR / filename).read_text()

    assert "/work/users/" not in script
    assert "Run from the repository root" in script
    assert "DRY_RUN=true" in script
    assert "sbatch" in script
    assert "PROJECT_ROOT" in script
    assert "SLURM_SUBMIT_DIR" in script


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_ENTRYPOINTS]
)
def test_multimodel_entrypoints_explain_how_to_adapt_them_to_another_cluster(filename):
    script = (TRAIN_DIR / filename).read_text()

    assert "Adapting this script to another cluster" in script
    assert "SBATCH_PARTITION" in script
    assert "CONDA_ENV" in script
    assert "CUDA_MODULE" in script
    assert "ACTOR_MODEL_PATH" in script
    assert "WANDB_MODE=offline" in script


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_ENTRYPOINTS]
)
def test_multimodel_entrypoints_never_hard_code_a_semantic_eos_token_id(filename):
    script = (TRAIN_DIR / filename).read_text()
    settings = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))

    assert "SEMANTIC_EOS_TOKEN_IDS" not in settings
    for token_id in MULTIMODEL_HARD_CODED_EOS_IDS:
        assert token_id not in settings, f"{filename} hard-codes EOS id {token_id}"


def test_reward_forward_token_budget_tracks_ppo_budget_and_can_be_overridden():
    inherited = run_script(LAUNCHER, PPO_MAX_TOKEN_LEN_PER_GPU=8192)
    overridden = run_script(
        LAUNCHER,
        PPO_MAX_TOKEN_LEN_PER_GPU=8192,
        REWARD_MAX_TOKEN_LEN_PER_GPU=4096,
    )

    assert inherited.returncode == 0, inherited.stderr
    assert "ppo_max_token_len_per_gpu=8192" in inherited.stdout
    assert "reward_max_token_len_per_gpu=8192" in inherited.stdout
    assert overridden.returncode == 0, overridden.stderr
    assert "reward_max_token_len_per_gpu=4096" in overridden.stdout
