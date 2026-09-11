import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


EVAL_DIR = Path(__file__).resolve().parent
MODULE_PATH = EVAL_DIR / "opd_eval_curve.py"


def load_eval_curve_module():
    assert MODULE_PATH.exists(), "opd_eval_curve.py is not implemented"
    sys.path.insert(0, str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location("opd_eval_curve", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=f"{prompt}/completion-{idx}",
                        token_ids=(prompt_idx, idx, 151645),
                    )
                    for idx in range(sampling_params.kwargs["n"])
                ]
            )
            for prompt_idx, prompt in enumerate(prompts)
        ]


class RecordingTokenizer:
    eos_token_id = 151643
    chat_template = "{% if enable_thinking %}think{% endif %}"

    def convert_ids_to_tokens(self, token_id):
        return {151645: "<|im_end|>"}[token_id]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<|im_end|>": [151645]}[text]

    def apply_chat_template(self, messages, **kwargs):
        expected = {"tokenize": False, "add_generation_prompt": True}
        if "chat_template" in kwargs:
            expected["chat_template"] = "template without a switch"
        else:
            expected["enable_thinking"] = False
        assert kwargs == expected
        return f"chat:{messages[0]['content']}"


def test_eval_prompt_templates_match_train_prompt_content(tmp_path):
    eval_curve = load_eval_curve_module()
    data_path = tmp_path / "test.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "reward_model": {"ground_truth": "4"},
            }
        ]
    ).to_parquet(data_path)

    assert eval_curve.load_samples(data_path, "ttrl")[0]["prompt"] == (
        "What is 2+2? Please reason step by step, and put your final answer within \\boxed{}."
    )
    assert eval_curve.load_samples(data_path, "eopd")[0]["prompt"] == "What is 2+2?"
    assert eval_curve.load_samples(data_path, "dapo")[0]["prompt"] == (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        "What is 2+2?\n\nRemember to put your answer on its own line after \"Answer:\"."
    )


def test_eval_grader_understands_ttrl_and_dapo_answer_formats():
    eval_curve = load_eval_curve_module()

    assert eval_curve.grade_answer_verl("work\n\\boxed{4}", "4", answer_format="ttrl")
    assert eval_curve.grade_answer_verl("work\nAnswer: 4", "4", answer_format="dapo")
    assert not eval_curve.grade_answer_verl("work\n4", "4", answer_format="dapo")

def test_batched_generation_submits_the_full_prompt_shard_once_with_n_completions():
    eval_curve = load_eval_curve_module()
    generate = getattr(eval_curve, "generate_batched_requests", None)
    assert generate is not None, "batched vLLM generation is not implemented"
    llm = RecordingLLM()
    task_payloads = [
        {
            "name": "AIME24",
            "samples": [
                {"example_id": 0, "prompt": "p0", "answer": "a0"},
                {"example_id": 1, "prompt": "p1", "answer": "a1"},
            ],
        },
        {
            "name": "AMC23",
            "samples": [{"example_id": 0, "prompt": "p2", "answer": "a2"}],
        },
    ]

    results = generate(
        llm=llm,
        tokenizer=RecordingTokenizer(),
        sampling_params_cls=RecordingSamplingParams,
        task_payloads=task_payloads,
        enable_thinking=False,
        n=16,
        max_tokens=8192,
        temperature=0.7,
        top_p=0.95,
        seed=0,
        stop_token_ids=[151643],
        blocked_token_ids=None,
    )

    assert len(llm.calls) == 1
    prompts, sampling_params, use_tqdm = llm.calls[0]
    assert prompts == ["chat:p0", "chat:p1", "chat:p2"]
    assert sampling_params.kwargs == {
        "n": 16,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 8192,
        "seed": 0,
        "stop_token_ids": [151643],
    }
    assert use_tqdm is False
    assert len(results["AIME24"]) == 32
    assert len(results["AMC23"]) == 16
    assert [row["seed"] for row in results["AMC23"]] == list(range(16))
    assert results["AIME24"][0]["_response_token_ids"] == [0, 0, 151645]
    assert results["AMC23"][-1]["_response_token_ids"] == [2, 15, 151645]


def test_batched_generation_blocks_manifest_tokens_for_canonical_eos():
    eval_curve = load_eval_curve_module()
    llm = RecordingLLM()

    eval_curve.generate_batched_requests(
        llm=llm,
        tokenizer=RecordingTokenizer(),
        sampling_params_cls=RecordingSamplingParams,
        task_payloads=[
            {
                "name": "AIME24",
                "samples": [{"example_id": 0, "prompt": "p0", "answer": "a0"}],
            }
        ],
        enable_thinking=False,
        n=1,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        seed=0,
        stop_token_ids=[151643],
        blocked_token_ids=[151645],
    )

    sampling_params = llm.calls[0][1]
    assert sampling_params.kwargs["stop_token_ids"] == [151643]
    assert sampling_params.kwargs["bad_words"] == ["<|im_end|>"]


def test_external_template_is_passed_as_text_without_qwen_thinking_kwarg():
    eval_curve = load_eval_curve_module()
    llm = RecordingLLM()
    tokenizer = RecordingTokenizer()
    tokenizer.chat_template = "template without a switch"

    eval_curve.generate_batched_requests(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params_cls=RecordingSamplingParams,
        task_payloads=[{"name": "AIME24", "samples": [{"example_id": 0, "prompt": "p", "answer": "a"}]}],
        enable_thinking=False,
        n=1,
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        seed=0,
        stop_token_ids=[151643],
        blocked_token_ids=None,
        chat_template="template without a switch",
    )

    assert llm.calls[0][0] == ["chat:p"]


def test_stop_tokens_default_to_the_model_eos_instead_of_hardcoded_chat_tokens():
    eval_curve = load_eval_curve_module()

    assert eval_curve.resolve_stop_token_ids(RecordingTokenizer(), None) == [151643]


def test_explicit_stop_tokens_override_the_model_eos_for_an_aligned_eos_run():
    eval_curve = load_eval_curve_module()

    assert eval_curve.resolve_stop_token_ids(RecordingTokenizer(), [151643, 151645]) == [151643, 151645]


def test_prompt_sharding_balances_all_one_hundred_eval_prompts_across_four_engines():
    eval_curve = load_eval_curve_module()
    split = getattr(eval_curve, "split_task_payloads_by_prompt", None)
    assert split is not None, "four-engine prompt sharding is not implemented"
    task_payloads = [
        {"name": "AIME24", "samples": [{"example_id": i} for i in range(30)]},
        {"name": "AIME25", "samples": [{"example_id": i} for i in range(30)]},
        {"name": "AMC23", "samples": [{"example_id": i} for i in range(40)]},
    ]

    chunks = split(task_payloads, num_workers=4)

    assert [sum(len(task["samples"]) for task in chunk) for chunk in chunks] == [25, 25, 25, 25]
    assert sum(len(task["samples"]) for chunk in chunks for task in chunk) == 100


def test_generation_writer_keeps_jsonl_unchanged_and_saves_token_ids_in_npz(tmp_path):
    eval_curve = load_eval_curve_module()
    write_results = getattr(eval_curve, "write_generation_results", None)
    assert write_results is not None, "NPZ token ID persistence is not implemented"
    output_path = tmp_path / "aime24.jsonl"
    rows = [
        {
            "example_id": 0,
            "prompt": "p0",
            "answer": "a0",
            "seed": 0,
            "response": "r0",
            "_response_token_ids": [11, 12, 151645],
        },
        {
            "example_id": 1,
            "prompt": "p1",
            "answer": "a1",
            "seed": 0,
            "response": "r1",
            "_response_token_ids": [21],
        },
    ]

    write_results(output_path, rows)

    with output_path.open("r", encoding="utf-8") as f:
        saved_rows = [json.loads(line) for line in f]
    assert saved_rows == [
        {"example_id": 0, "prompt": "p0", "answer": "a0", "seed": 0, "response": "r0"},
        {"example_id": 1, "prompt": "p1", "answer": "a1", "seed": 0, "response": "r1"},
    ]

    with np.load(output_path.with_suffix(".tokens.npz"), allow_pickle=False) as token_data:
        np.testing.assert_array_equal(token_data["token_ids"], np.array([11, 12, 151645, 21], dtype=np.int32))
        np.testing.assert_array_equal(token_data["offsets"], np.array([0, 3, 4], dtype=np.int64))


def test_generation_output_is_reusable_only_with_matching_token_npz(tmp_path):
    eval_curve = load_eval_curve_module()
    output_path = tmp_path / "aime24.jsonl"
    rows = [
        {
            "example_id": 0,
            "prompt": "p0",
            "answer": "a0",
            "seed": 0,
            "response": "r0",
            "_response_token_ids": [11, 151643],
        }
    ]

    output_path.write_text(json.dumps({"example_id": 0}) + "\n", encoding="utf-8")
    assert not eval_curve.has_complete_generation_output(output_path, expected_rows=1)

    output_path.with_suffix(".tokens.npz").write_bytes(b"interrupted")
    assert not eval_curve.has_complete_generation_output(output_path, expected_rows=1)

    eval_curve.write_generation_results(output_path, rows)
    assert eval_curve.has_complete_generation_output(output_path, expected_rows=1)

    np.savez(
        output_path.with_suffix(".tokens.npz"),
        token_ids=np.array([11, 151643], dtype=np.int32),
        offsets=np.array([0, 3], dtype=np.int64),
    )
    assert not eval_curve.has_complete_generation_output(output_path, expected_rows=1)


def test_eval_output_manifest_rejects_reuse_under_different_prompt_or_eos_semantics(tmp_path):
    eval_curve = load_eval_curve_module()
    baseline_config = {
        "schema_version": 1,
        "prompt_template": "eopd",
        "enable_thinking": False,
        "stop_token_ids": [151643],
        "blocked_token_ids": [],
        "n": 16,
        "max_tokens": 8192,
    }

    eval_curve.prepare_eval_output_root(tmp_path, baseline_config, replace=False)
    assert json.loads((tmp_path / "eval_run_config.json").read_text()) == baseline_config

    ttrl_config = {**baseline_config, "prompt_template": "ttrl"}
    with pytest.raises(RuntimeError, match="different evaluation settings"):
        eval_curve.prepare_eval_output_root(tmp_path, ttrl_config, replace=False)

    canonical_config = {**baseline_config, "blocked_token_ids": [151645]}
    with pytest.raises(RuntimeError, match="different evaluation settings"):
        eval_curve.prepare_eval_output_root(tmp_path, canonical_config, replace=False)

    eval_curve.prepare_eval_output_root(tmp_path, ttrl_config, replace=True)
    assert json.loads((tmp_path / "eval_run_config.json").read_text()) == ttrl_config


def test_grading_reports_semantic_format_length_and_token_finish_metrics(tmp_path):
    eval_curve = load_eval_curve_module()
    output_path = tmp_path / "aime24.jsonl"
    eval_curve.write_generation_results(
        output_path,
        [
            {
                "example_id": 0,
                "prompt": "p",
                "answer": "4",
                "seed": 0,
                "response": "work\n\\boxed{4}",
                "_response_token_ids": [7, 151643],
            },
            {
                "example_id": 0,
                "prompt": "p",
                "answer": "4",
                "seed": 1,
                "response": "work\nAnswer: 4",
                "_response_token_ids": [7, 151645],
            },
            {
                "example_id": 0,
                "prompt": "p",
                "answer": "4",
                "seed": 2,
                "response": "no extractable answer",
                "_response_token_ids": [7, 8],
            },
        ],
    )

    metrics = eval_curve.grade_file(
        output_path,
        length_tokenizer=None,
        prompt_template="ttrl",
        stop_token_ids=[151643, 151645],
        max_tokens=2,
    )

    assert metrics["semantic_accuracy"] == pytest.approx(2 / 3)
    assert metrics["extraction_success"] == pytest.approx(2 / 3)
    assert metrics["format_compliance"] == pytest.approx(1 / 3)
    assert metrics["output_length_mean"] == pytest.approx(2.0)
    assert metrics["output_length_median"] == pytest.approx(2.0)
    assert metrics["output_length_p95"] == pytest.approx(2.0)
    assert metrics["finish_e1_fraction"] == pytest.approx(1 / 3)
    assert metrics["finish_e2_fraction"] == pytest.approx(1 / 3)
    assert metrics["truncation_rate"] == pytest.approx(1 / 3)


def test_grading_reports_each_finish_id_for_three_terminal_tokens(tmp_path):
    eval_curve = load_eval_curve_module()
    output_path = tmp_path / "aime24.jsonl"
    eval_curve.write_generation_results(
        output_path,
        [
            {
                "example_id": index,
                "prompt": "p",
                "answer": "4",
                "seed": 0,
                "response": "\\boxed{4}",
                "_response_token_ids": [7, terminal_id],
            }
            for index, terminal_id in enumerate((11, 13, 17))
        ],
    )

    metrics = eval_curve.grade_file(
        output_path,
        length_tokenizer=None,
        prompt_template="ttrl",
        stop_token_ids=[11, 13, 17],
        max_tokens=8,
    )

    assert metrics["finish_token_11_fraction"] == pytest.approx(1 / 3)
    assert metrics["finish_token_13_fraction"] == pytest.approx(1 / 3)
    assert metrics["finish_token_17_fraction"] == pytest.approx(1 / 3)
    assert "finish_e1_fraction" not in metrics
