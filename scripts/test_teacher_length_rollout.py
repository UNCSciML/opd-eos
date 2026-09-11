import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENTRYPOINT = PROJECT_ROOT / "scripts" / "infer" / "teacher_length_rollout.py"
SLURM_ENTRYPOINT = (
    PROJECT_ROOT / "slurm" / "diagnostic" / "measure_teacher_lengths_4x6000pro.sl"
)


def load_module():
    spec = importlib.util.spec_from_file_location("teacher_length_rollout", PYTHON_ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_prompt_batch_uses_training_sampling_and_vllm_token_lengths():
    module = load_module()

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, chat, **kwargs):
            self.calls.append((chat, kwargs))
            return f"formatted-{chat[0]['content']}"

    class CheckingSamplingParams:
        def __init__(self, **kwargs):
            assert kwargs == {
                "n": 2,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "repetition_penalty": 1.0,
                "max_tokens": 7168,
                "stop_token_ids": [151645, 151643],
            }

    class RecordingLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, sampling, use_tqdm):
            self.calls.append((prompts, sampling, use_tqdm))
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[10, 11, 151645], finish_reason="stop", stop_reason=151645
                        ),
                        SimpleNamespace(
                            token_ids=[20, 21, 22, 23], finish_reason="length", stop_reason=None
                        ),
                    ]
                )
            ]

    tokenizer = RecordingTokenizer()
    llm = RecordingLLM()
    rows = module.generate_prompt_batch(
        indexed_chats=[(7, [{"role": "user", "content": "question"}])],
        llm=llm,
        tokenizer=tokenizer,
        sampling_params_cls=CheckingSamplingParams,
        n=2,
        max_tokens=7168,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        stop_token_ids=[151645, 151643],
        enable_thinking=False,
    )

    assert llm.calls[0][0] == ["formatted-question"]
    assert llm.calls[0][2] is True
    assert tokenizer.calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert rows == [
        {
            "prompt_index": 7,
            "rollout_index": 0,
            "response_length": 3,
            "finish_reason": "stop",
            "stop_reason": "151645",
        },
        {
            "prompt_index": 7,
            "rollout_index": 1,
            "response_length": 4,
            "finish_reason": "length",
            "stop_reason": "",
        },
    ]


def test_summary_reports_mean_and_finish_reason_fractions():
    module = load_module()
    rows = [
        {"response_length": 1, "finish_reason": "stop", "stop_reason": "151645"},
        {"response_length": 3, "finish_reason": "stop", "stop_reason": "151643"},
        {"response_length": 8, "finish_reason": "length", "stop_reason": ""},
    ]

    summary = module.summarize_rows(rows, max_tokens=8)

    assert summary["num_generations"] == 3
    assert summary["response_length_mean"] == 4.0
    assert summary["response_length_min"] == 1
    assert summary["response_length_max"] == 8
    assert summary["finish_reason_counts"] == {"length": 1, "stop": 2}
    assert summary["stop_reason_counts"] == {"151643": 1, "151645": 1}
    assert summary["length_capped_fraction"] == 1 / 3


def test_slurm_dry_run_matches_the_200_step_training_prompt_stream():
    result = subprocess.run(
        ["bash", str(SLURM_ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "DRY_RUN": "true", "SLURM_JOB_ID": "unit-test"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "max_prompts=3200" in result.stdout
    assert "n=4" in result.stdout
    assert "max_tokens=7168" in result.stdout
    assert "max_model_len=8192" in result.stdout
    assert "temperature=1.0" in result.stdout
    assert "top_p=1.0" in result.stdout
    assert "top_k=-1" in result.stdout
    assert "stop_token_ids=151645,151643" in result.stdout
    assert "enable_thinking=false" in result.stdout
    assert "tensor_parallel_size=1" in result.stdout
    assert "num_workers=4" in result.stdout
