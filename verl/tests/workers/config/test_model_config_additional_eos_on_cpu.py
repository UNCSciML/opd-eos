from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoConfig, GenerationConfig, Qwen3Config

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig


def _write_qwen_config(model_dir):
    Qwen3Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        architectures=["Qwen3ForCausalLM"],
        eos_token_id=151643,
        pad_token_id=151643,
    ).save_pretrained(model_dir)
    GenerationConfig(eos_token_id=151643, pad_token_id=151643).save_pretrained(model_dir)


def test_additional_eos_ids_patch_runtime_configs_without_mutating_source_files(tmp_path):
    _write_qwen_config(tmp_path)

    model_config = HFModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        additional_eos_token_ids=[151645, 151643],
    )

    assert model_config.effective_eos_token_ids == [151643, 151645]
    assert model_config.hf_config.eos_token_id == [151643, 151645]
    assert model_config.generation_config.eos_token_id == [151643, 151645]
    assert AutoConfig.from_pretrained(tmp_path).eos_token_id == 151643
    assert GenerationConfig.from_pretrained(tmp_path).eos_token_id == 151643


def test_default_model_config_preserves_the_original_single_eos(tmp_path):
    _write_qwen_config(tmp_path)

    model_config = HFModelConfig(path=str(tmp_path), load_tokenizer=False)

    assert model_config.effective_eos_token_ids == 151643
    assert model_config.hf_config.eos_token_id == 151643
    assert model_config.generation_config.eos_token_id == 151643


class RecordingLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        return [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7, 151645, 8], logprobs=[])])
            for _ in prompts
        ]


class RecordingSamplingParams:
    def __init__(self, **kwargs):
        defaults = {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "ignore_eos": False,
            "stop_token_ids": None,
        }
        for name, value in defaults.items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs


def test_vllm_rollout_stops_on_the_effective_student_eos_ids(monkeypatch):
    from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd

    monkeypatch.setattr(vllm_rollout_spmd, "LLM", RecordingLLM)
    monkeypatch.setattr(vllm_rollout_spmd, "SamplingParams", RecordingSamplingParams)
    monkeypatch.setattr(vllm_rollout_spmd.torch.distributed, "get_world_size", lambda: 1)

    rollout_config = RolloutConfig(
        name="vllm",
        prompt_length=32,
        response_length=64,
        max_model_len=96,
        max_num_batched_tokens=96,
        tensor_model_parallel_size=1,
        load_format="dummy",
    )
    model_config = SimpleNamespace(
        local_path="/unused/model",
        tokenizer=SimpleNamespace(pad_token_id=151643),
        hf_config=SimpleNamespace(rope_scaling=None, max_position_embeddings=4096),
        trust_remote_code=False,
        lora_adapter_path=None,
        lora_rank=0,
        additional_eos_token_ids=[151645],
        effective_eos_token_ids=[151643, 151645],
    )

    rollout = vllm_rollout_spmd.vLLMRollout(rollout_config, model_config, device_mesh=None)

    assert rollout.sampling_params.stop_token_ids == [151643, 151645]


def test_vllm_response_mask_uses_the_effective_student_eos_ids(monkeypatch):
    from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd

    monkeypatch.setattr(vllm_rollout_spmd, "LLM", RecordingLLM)
    monkeypatch.setattr(vllm_rollout_spmd, "SamplingParams", RecordingSamplingParams)
    monkeypatch.setattr(vllm_rollout_spmd.torch.distributed, "get_world_size", lambda: 1)

    rollout_config = RolloutConfig(
        name="vllm",
        prompt_length=2,
        response_length=4,
        max_model_len=6,
        max_num_batched_tokens=6,
        tensor_model_parallel_size=1,
        load_format="dummy",
    )
    model_config = SimpleNamespace(
        local_path="/unused/model",
        tokenizer=SimpleNamespace(pad_token_id=151643),
        hf_config=SimpleNamespace(rope_scaling=None, max_position_embeddings=4096),
        trust_remote_code=False,
        lora_adapter_path=None,
        lora_rank=0,
        additional_eos_token_ids=[151645],
        effective_eos_token_ids=[151643, 151645],
    )
    rollout = vllm_rollout_spmd.vLLMRollout(rollout_config, model_config, device_mesh=None)
    prompts = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]]),
        },
        non_tensors={"raw_prompt_ids": np.array([[11, 12]], dtype=object)},
        meta_info={"eos_token_id": 151643, "pad_token_id": 151643},
    )

    output = rollout.generate_sequences(prompts)

    torch.testing.assert_close(output.batch["responses"], torch.tensor([[7, 151645, 8, 151643]]))
    torch.testing.assert_close(output.batch["attention_mask"][:, -4:], torch.tensor([[1, 1, 0, 0]]))
