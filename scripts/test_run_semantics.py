import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parent / "run_semantics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_semantics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_run_semantics_records_the_effective_train_eos_and_nothinking():
    module = load_module()

    semantics = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=151643,
        additional_eos_token_ids=[],
        enable_thinking=False,
        eos_mode="baseline",
        semantic_eos_token_ids=[151643, 151645],
        eos_mapping_epsilon=1e-12,
        prompt_template="ttrl",
        train_dataset="/data/ttrl.parquet",
    )

    assert semantics == {
        "schema_version": 3,
        "model_path": "/models/student",
        "student_model_path": "/models/student",
        "teacher_model_path": None,
        "chat_template_source": "/models/student",
        "chat_template_sha256": None,
        "thinking_supported": None,
        "enable_thinking": False,
        "prompt_template": "ttrl",
        "train_dataset": "/data/ttrl.parquet",
        "eos_mode": "baseline",
        "semantic_terminal_token_ids": [151643, 151645],
        "semantic_eos_token_ids": [151643, 151645],
        "eos_mapping_epsilon": 1e-12,
        "student_native_eos_token_ids": [151643],
        "teacher_native_eos_token_ids": [],
        "model_eos_token_ids": [151643],
        "additional_eos_token_ids": [],
        "effective_eos_token_ids": [151643],
        "rollout_stop_token_ids": [151643],
        "blocked_token_ids": [],
        "tracked_token_ids": [151643, 151645],
        "tracked_token_names": ["token_151643", "token_151645"],
    }


def test_build_run_semantics_records_both_ids_only_for_an_aligned_eos_run():
    module = load_module()

    semantics = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=151643,
        additional_eos_token_ids=[151645, 151643],
        enable_thinking=False,
        eos_mode="two_stop",
        semantic_eos_token_ids=[151643, 151645],
        eos_mapping_epsilon=1e-12,
        prompt_template="eopd",
        train_dataset="/data/raw.parquet",
    )

    assert semantics["effective_eos_token_ids"] == [151643, 151645]
    assert semantics["rollout_stop_token_ids"] == [151643, 151645]


def test_build_run_semantics_uses_ordered_student_teacher_union_for_k_token_modes():
    module = load_module()

    baseline = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=[1, 106],
        teacher_model_path="/models/teacher",
        teacher_eos_token_ids=[106, 128],
        additional_eos_token_ids=[],
        enable_thinking=False,
        eos_mode="baseline",
        semantic_eos_token_ids=[],
        eos_mapping_epsilon=1e-12,
        prompt_template="ttrl",
        train_dataset="/data/ttrl.parquet",
    )
    semantic = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=[1, 106],
        teacher_model_path="/models/teacher",
        teacher_eos_token_ids=[106, 128],
        additional_eos_token_ids=[],
        enable_thinking=False,
        eos_mode="semantic_class",
        semantic_eos_token_ids=[],
        eos_mapping_epsilon=1e-12,
        prompt_template="ttrl",
        train_dataset="/data/ttrl.parquet",
    )

    assert baseline["student_native_eos_token_ids"] == [1, 106]
    assert baseline["teacher_native_eos_token_ids"] == [106, 128]
    assert baseline["semantic_terminal_token_ids"] == [1, 106, 128]
    assert baseline["rollout_stop_token_ids"] == [1, 106]
    assert baseline["additional_eos_token_ids"] == []
    assert semantic["rollout_stop_token_ids"] == [1, 106, 128]
    assert semantic["additional_eos_token_ids"] == [128]
    assert semantic["effective_eos_token_ids"] == [1, 106, 128]


@pytest.mark.parametrize(
    ("student_ids", "teacher_ids", "canonical_id"),
    [
        ([1, 106], [1, 106], 1),
        ([128001], [128001, 128008, 128009], 128001),
        ([1, 106], [1, 106], 106),
    ],
)
@pytest.mark.parametrize("mode", ["teacher_map", "canonical"])
def test_mapping_manifest_uses_full_semantic_set_and_one_canonical_stop(student_ids, teacher_ids, canonical_id, mode):
    module = load_module()
    semantics = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=student_ids,
        teacher_eos_token_ids=teacher_ids,
        additional_eos_token_ids=[],
        enable_thinking=False,
        eos_mode=mode,
        semantic_eos_token_ids=[],
        canonical_eos_token_id=canonical_id,
        eos_mapping_epsilon=1e-12,
        prompt_template="ttrl",
        train_dataset="/data/ttrl.parquet",
    )

    other_ids = [token_id for token_id in teacher_ids if token_id != canonical_id]
    assert semantics["semantic_terminal_token_ids"] == [canonical_id, *other_ids]
    assert semantics["canonical_eos_token_id"] == canonical_id
    assert semantics["rollout_stop_token_ids"] == [canonical_id]
    assert semantics["blocked_token_ids"] == (other_ids if mode == "canonical" else [])


def test_eos_discovery_prefers_generation_config(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        "transformers.GenerationConfig.from_pretrained",
        lambda *args, **kwargs: type("Generation", (), {"eos_token_id": [1, 106]})(),
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: pytest.fail("AutoConfig should not be loaded when generation_config has EOS ids"),
    )

    assert module.discover_model_eos_token_ids("/models/student") == [1, 106]


def test_canonical_manifest_records_e2_as_blocked_but_not_terminal():
    module = load_module()
    semantics = module.build_run_semantics(
        model_path="/models/student",
        model_eos_token_ids=151643,
        additional_eos_token_ids=[],
        enable_thinking=False,
        eos_mode="canonical",
        semantic_eos_token_ids=[151643, 151645],
        eos_mapping_epsilon=1e-12,
        prompt_template="eopd",
        train_dataset="/data/raw.parquet",
    )

    assert semantics["rollout_stop_token_ids"] == [151643]
    assert semantics["blocked_token_ids"] == [151645]


def test_get_cli_prints_manifest_values_for_the_eval_watcher(tmp_path):
    manifest = tmp_path / "run_semantics.json"
    manifest.write_text(
        json.dumps(
            {
                "enable_thinking": False,
                "effective_eos_token_ids": [151643, 151645],
            }
        ),
        encoding="utf-8",
    )

    eos = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "get", "--path", str(manifest), "--field", "effective_eos_token_ids"],
        check=True,
        capture_output=True,
        text=True,
    )
    thinking = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "get", "--path", str(manifest), "--field", "enable_thinking"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert eos.stdout.strip() == "151643,151645"
    assert thinking.stdout.strip() == "false"
