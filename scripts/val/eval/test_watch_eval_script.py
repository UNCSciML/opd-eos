import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WATCHER = PROJECT_ROOT / "watch_eval_opd_sampled_token_qwen3_1p7b_ckpts_batched_n16_4l40s.sl"


def test_watcher_dry_run_uses_parameterized_paths_and_saved_train_semantics(tmp_path):
    checkpoint_root = tmp_path / "checkpoint" / "new_run"
    base_model = "Qwen/Qwen3-1.7B-Base"
    output_root = tmp_path / "eval_output"
    merged_root = tmp_path / "merged"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "run_semantics.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_path": str(base_model),
                "enable_thinking": False,
                "prompt_template": "eopd",
                "eos_mode": "canonical",
                "model_eos_token_ids": [151643],
                "additional_eos_token_ids": [],
                "effective_eos_token_ids": [151643],
                "rollout_stop_token_ids": [151643],
                "blocked_token_ids": [151645],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(WATCHER),
            "--dry-run",
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-root",
            str(output_root),
            "--merged-root",
            str(merged_root),
            "--conda-env",
            sys.executable.rsplit("/bin/python", 1)[0],
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"checkpoint_root={checkpoint_root}" in result.stdout
    assert f"base_model={base_model}" in result.stdout
    assert f"output_root={output_root}" in result.stdout
    assert f"merged_root={merged_root}" in result.stdout
    assert "enable_thinking=false" in result.stdout
    assert "stop_token_ids=151643" in result.stdout
    assert "blocked_token_ids=151645" in result.stdout
    assert "prompt_template=eopd" in result.stdout
    assert "eval_templates=eopd" in result.stdout
    assert "66118896" not in result.stdout


def test_watcher_rejects_legacy_manifest_missing_exact_stop_and_template_fields(tmp_path):
    checkpoint_root = tmp_path / "checkpoint" / "legacy_run"
    base_model = tmp_path / "base_model"
    checkpoint_root.mkdir(parents=True)
    base_model.mkdir()
    (checkpoint_root / "run_semantics.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_path": str(base_model),
                "enable_thinking": False,
                "effective_eos_token_ids": [151643],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(WATCHER),
            "--dry-run",
            "--checkpoint-root",
            str(checkpoint_root),
            "--conda-env",
            sys.executable.rsplit("/bin/python", 1)[0],
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "rollout_stop_token_ids" in result.stderr
