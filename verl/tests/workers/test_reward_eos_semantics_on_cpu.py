from types import SimpleNamespace

import torch
from torch import nn

from verl.utils.eos_semantics import map_teacher_sampled_log_probs
from verl.workers.fsdp_workers import RewardModelWorker


class FixedTeacher(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.fixed_logits.clone())


def test_teacher_forward_returns_raw_eos_probabilities_and_applies_mapping(monkeypatch):
    monkeypatch.setattr("verl.workers.fsdp_workers.get_device_name", lambda: "cpu")
    monkeypatch.setattr(
        "verl.utils.torch_functional.logprobs_from_logits",
        lambda logits, labels, **_: torch.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1),
    )
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    worker = RewardModelWorker.__new__(RewardModelWorker)
    worker.reward_module = FixedTeacher(logits)
    worker.use_remove_padding = False
    worker.use_fused_kernels = False
    responses = torch.tensor([[2, 3]])
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": responses,
    }

    outputs = worker._forward_micro_batch(
        micro_batch,
        eos_mode="teacher_map",
        semantic_eos_token_ids=[2, 4],
        eos_mapping_epsilon=1e-4,
    )
    sampled_log_probs, teacher_eos_log_probs = outputs[0], outputs[-1]

    response_log_probs = torch.log_softmax(logits[:, 1:3, :], dim=-1)
    expected_eos = response_log_probs[..., [2, 4]]
    expected_sampled = response_log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    expected_sampled = map_teacher_sampled_log_probs(
        raw_sampled_log_probs=expected_sampled,
        teacher_eos_log_probs=expected_eos,
        sampled_ids=responses,
        mode="teacher_map",
        semantic_eos_token_ids=[2, 4],
        epsilon=1e-4,
    )
    torch.testing.assert_close(teacher_eos_log_probs, expected_eos)
    torch.testing.assert_close(sampled_log_probs, expected_sampled)


def test_teacher_forward_semantic_class_aggregates_three_terminal_tokens(monkeypatch):
    monkeypatch.setattr("verl.workers.fsdp_workers.get_device_name", lambda: "cpu")
    monkeypatch.setattr(
        "verl.utils.torch_functional.logprobs_from_logits",
        lambda logits, labels, **_: torch.log_softmax(logits, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1),
    )
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    worker = RewardModelWorker.__new__(RewardModelWorker)
    worker.reward_module = FixedTeacher(logits)
    worker.use_remove_padding = False
    worker.use_fused_kernels = False
    responses = torch.tensor([[2, 3]])
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": responses,
    }

    outputs = worker._forward_micro_batch(
        micro_batch,
        eos_mode="semantic_class",
        semantic_eos_token_ids=[2, 4, 5],
    )
    sampled_log_probs, teacher_eos_log_probs = outputs[0], outputs[-1]

    response_log_probs = torch.log_softmax(logits[:, 1:3, :], dim=-1)
    expected_eos = response_log_probs[..., [2, 4, 5]]
    expected_sampled = response_log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    expected_sampled[:, 0] = torch.logsumexp(expected_eos[:, 0], dim=-1)
    torch.testing.assert_close(teacher_eos_log_probs, expected_eos)
    torch.testing.assert_close(sampled_log_probs, expected_sampled)
