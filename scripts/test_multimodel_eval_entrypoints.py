import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "slurm" / "eval"
SHARED_CHECKPOINT_EVAL = PROJECT_ROOT / "eval_opd_ttrl_qwen3_1p7b_ckpts20_200_batched_n16_4gpu.sl"
MULTIMODEL_SUBMIT = PROJECT_ROOT / "slurm" / "submit_multimodel_runs.sh"

MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS = [
    (
        "eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        [128001],
    ),
    (
        "eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl",
        "google/gemma-3-4b-pt",
        "google/gemma-3-4b-it",
        [1, 106],
    ),
]


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


def write_manifest(
    tmp_path: Path,
    student: str,
    teacher: str,
    stop_ids: list[int],
    eos_mode: str = "baseline",
) -> Path:
    """Write the manifest a TTRL train run leaves in its checkpoint root."""
    checkpoint_root = tmp_path / "checkpoint" / "example-multimodel-run"
    checkpoint_root.mkdir(parents=True)
    semantics = {
        "schema_version": 3,
        "model_path": student,
        "student_model_path": student,
        "teacher_model_path": teacher,
        "chat_template_source": teacher,
        "enable_thinking": False,
        "prompt_template": "ttrl",
        "eos_mode": eos_mode,
        "semantic_terminal_token_ids": stop_ids,
        "rollout_stop_token_ids": stop_ids,
        "blocked_token_ids": [],
    }
    (checkpoint_root / "run_semantics.json").write_text(json.dumps(semantics))
    return checkpoint_root


def resolved_config(stdout: str) -> dict[str, str]:
    config = {}
    for line in stdout.splitlines():
        if "=" in line and not line.startswith("+"):
            key, _, value = line.partition("=")
            config[key] = value
    return config


@pytest.mark.parametrize(
    ("filename", "student", "teacher", "stop_ids"), MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS
)
def test_multimodel_checkpoint_eval_inherits_every_setting_from_the_train_manifest(
    tmp_path, filename, student, teacher, stop_ids
):
    checkpoint_root = write_manifest(tmp_path, student, teacher, stop_ids)

    result = run_script(
        EVAL_DIR / filename,
        CHECKPOINT_ROOT=checkpoint_root,
        CONDA_ENV=Path(sys.executable).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    config = resolved_config(result.stdout)
    assert config["project_root"] == str(PROJECT_ROOT)
    assert config["checkpoint_root"] == str(checkpoint_root)
    assert config["base_model"] == student
    assert config["chat_template_model"] == teacher
    assert config["eos_mode"] == "baseline"
    assert config["stop_token_ids"] == ",".join(str(token_id) for token_id in stop_ids)
    assert config["blocked_token_ids"] == ""
    assert config["enable_thinking"] == "false"
    assert config["eval_template"] == "ttrl"
    assert config["tasks"] == "AIME24,AIME25,AMC23"
    assert config["n"] == "16"
    assert config["max_tokens"] == "8192"
    assert config["save_token_ids"] == "true"
    assert config["wandb_project"] == "OPD Length Inflation"


@pytest.mark.parametrize(
    ("filename", "student", "teacher", "stop_ids"), MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS
)
def test_multimodel_checkpoint_eval_resolves_exactly_the_shared_checkpoint_eval_config(
    tmp_path, filename, student, teacher, stop_ids
):
    checkpoint_root = write_manifest(tmp_path, student, teacher, stop_ids)
    conda_env = Path(sys.executable).resolve().parent.parent

    wrapper = run_script(EVAL_DIR / filename, CHECKPOINT_ROOT=checkpoint_root, CONDA_ENV=conda_env)
    shared = run_script(SHARED_CHECKPOINT_EVAL, CHECKPOINT_ROOT=checkpoint_root, CONDA_ENV=conda_env)

    assert wrapper.returncode == 0, wrapper.stderr
    assert shared.returncode == 0, shared.stderr
    assert resolved_config(wrapper.stdout) == resolved_config(shared.stdout)


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS]
)
def test_multimodel_checkpoint_eval_requires_an_explicit_checkpoint_root(filename):
    result = run_script(EVAL_DIR / filename)

    assert result.returncode == 2
    assert "CHECKPOINT_ROOT" in result.stderr


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS]
)
def test_multimodel_checkpoint_eval_entrypoints_are_portable_and_document_submission(filename):
    script = (EVAL_DIR / filename).read_text()

    assert "/work/users/" not in script
    assert "Run from the repository root" in script
    assert "DRY_RUN=true" in script
    assert "sbatch" in script
    assert "PROJECT_ROOT" in script
    assert "SLURM_SUBMIT_DIR" in script


@pytest.mark.parametrize(
    ("filename", "student", "teacher", "stop_ids"), MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS
)
def test_multimodel_checkpoint_eval_discovers_project_root_after_repository_is_moved(
    tmp_path, filename, student, teacher, stop_ids
):
    moved_root = tmp_path / "renamed-repository"
    shutil.copytree(PROJECT_ROOT / "slurm" / "eval", moved_root / "slurm" / "eval")
    shutil.copy2(SHARED_CHECKPOINT_EVAL, moved_root / SHARED_CHECKPOINT_EVAL.name)
    (moved_root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "run_semantics.py", moved_root / "scripts" / "run_semantics.py")
    checkpoint_root = write_manifest(tmp_path, student, teacher, stop_ids)

    result = run_script(
        moved_root / "slurm" / "eval" / filename,
        CHECKPOINT_ROOT=checkpoint_root,
        CONDA_ENV=Path(sys.executable).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert f"project_root={moved_root}" in result.stdout


def test_multimodel_checkpoint_eval_serves_every_eos_mode_of_its_model_pair(tmp_path):
    """One eval file per pair: the EOS mode is read from the run, never chosen here."""
    student, teacher, stop_ids = "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", [128001, 128008, 128009]
    conda_env = Path(sys.executable).resolve().parent.parent
    entrypoint = EVAL_DIR / "eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl"

    resolved = {}
    for mode in ("baseline", "semantic_class"):
        checkpoint_root = write_manifest(tmp_path / mode, student, teacher, stop_ids, eos_mode=mode)
        result = run_script(entrypoint, CHECKPOINT_ROOT=checkpoint_root, CONDA_ENV=conda_env)
        assert result.returncode == 0, result.stderr
        resolved[mode] = resolved_config(result.stdout)

    assert resolved["baseline"]["eos_mode"] == "baseline"
    assert resolved["semantic_class"]["eos_mode"] == "semantic_class"
    assert resolved["baseline"]["output_root"] != resolved["semantic_class"]["output_root"]


@pytest.mark.parametrize(
    "filename", [entrypoint[0] for entrypoint in MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS]
)
def test_multimodel_checkpoint_eval_explains_how_to_adapt_it_to_another_cluster(filename):
    script = (EVAL_DIR / filename).read_text()

    assert "Adapting this script to another cluster" in script
    assert "SBATCH_PARTITION" in script
    assert "CONDA_ENV" in script
    assert "CUDA_MODULE" in script
    assert "MERGED_ROOT" in script
    assert "WANDB_MODE=offline" in script


def test_multimodel_submit_helper_chains_one_matching_eval_after_each_train():
    result = run_script(MULTIMODEL_SUBMIT, RUN_TAG="test-multimodel")

    assert result.returncode == 0, result.stderr
    expected = {
        ("llama", "baseline"): "opd_ttrl_llama32_3b_base_to_instruct_baseline_test-multimodel",
        ("llama", "semantic_class"): "opd_ttrl_llama32_3b_base_to_instruct_semantic_class_test-multimodel",
        ("gemma", "baseline"): "opd_ttrl_gemma3_4bpt_to_4bit_baseline_test-multimodel",
        ("gemma", "semantic_class"): "opd_ttrl_gemma3_4bpt_to_4bit_semantic_class_test-multimodel",
    }
    for (model, mode), experiment in expected.items():
        assert f"model={model} mode={mode} train_job=DRY_RUN-{model}-{mode}" in result.stdout
        assert f"experiment_name={experiment}" in result.stdout
        assert f"checkpoint/{experiment}" in result.stdout
        assert f"dependency=afterok:DRY_RUN-{model}-{mode}" in result.stdout
    assert result.stdout.count("eval_job=DRY_RUN") == 4


@pytest.mark.parametrize(
    ("selector", "expected_jobs"),
    [
        ({"MODELS": "gemma"}, 2),
        ({"EOS_MODES": "semantic_class"}, 2),
        ({"MODELS": "llama", "EOS_MODES": "baseline"}, 1),
    ],
)
def test_multimodel_submit_helper_can_launch_a_subset(selector, expected_jobs):
    result = run_script(MULTIMODEL_SUBMIT, RUN_TAG="test-subset", **selector)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("eval_job=DRY_RUN") == expected_jobs
    if selector.get("MODELS") == "gemma":
        assert "model=llama" not in result.stdout
    if selector.get("EOS_MODES") == "semantic_class":
        assert "mode=baseline" not in result.stdout


def test_multimodel_submit_helper_rejects_an_unknown_selection():
    result = run_script(MULTIMODEL_SUBMIT, MODELS="mistral")

    assert result.returncode == 2
    assert "llama,gemma" in result.stderr


def test_multimodel_entrypoints_lead_with_the_disk_requirement(tmp_path):
    """Disk is the first thing that bites a collaborator, so it must come first."""
    scripts = [EVAL_DIR / name for name, *_ in MULTIMODEL_BASELINE_EVAL_ENTRYPOINTS]
    scripts.append(MULTIMODEL_SUBMIT)
    scripts += sorted((PROJECT_ROOT / "slurm" / "train").glob("train_ttrl_llama32_3b_*.sl"))
    scripts += sorted((PROJECT_ROOT / "slurm" / "train").glob("train_ttrl_gemma3_4bpt_*.sl"))

    for script in scripts:
        prose = [
            line for line in script.read_text().splitlines()
            if line.startswith("#") and not line.startswith("#!") and not line.startswith("#SBATCH")
        ]
        assert prose, script
        # At most a one-line title may precede it; no other section may.
        head = "\n".join(prose[:3])
        assert "DISK REQUIREMENTS" in head, f"{script.name} does not lead with disk requirements"
        assert any("checkpoint_merged/" in line for line in prose[:14]), script.name


def test_multimodel_submit_helper_explains_how_to_adapt_it_to_another_cluster():
    script = MULTIMODEL_SUBMIT.read_text()

    assert "Adapting this script to another cluster" in script
    assert "SBATCH_PARTITION" in script
    assert "CONDA_ENV" in script
    assert "DRY_RUN=true" in script
    # The Gemma caveat must stay visible and must not oversell semantic_class:
    # the terminal sets match, but there is almost no termination mass to aggregate,
    # so the run is a control separating two mechanisms, not an expected fix.
    assert "terminal SET already matches" in script
    assert "NOT expected to fix" in script
    assert "distribution-shift failure" in script
    assert "no mismatch" not in script.lower()
