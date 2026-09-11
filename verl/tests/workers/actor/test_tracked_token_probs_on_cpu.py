# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor


class FixedLogitsModel(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=self.fixed_logits.clone())


class ConfigNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_update_policy_passes_expected_eos_advantages_to_policy_loss(monkeypatch):
    captured = {}
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = nn.Linear(2, 1, bias=False)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor.ulysses_sequence_parallel_size = 1
    actor.config = ConfigNamespace(
        use_kl_loss=False,
        ppo_mini_batch_size=1,
        ppo_epochs=1,
        use_dynamic_bsz=False,
        ppo_micro_batch_size_per_gpu=1,
        entropy_coeff=0,
        loss_agg_mode="token-mean",
        policy_loss={"loss_mode": "vanilla"},
    )

    def fake_forward_micro_batch(*args, **kwargs):
        log_probs = actor.actor_module.weight.reshape(1, 2)
        return None, log_probs, None, None, None

    def fake_policy_loss(**kwargs):
        captured["advantages"] = kwargs["advantages"].detach().clone()
        return kwargs["log_prob"].sum(), {}

    actor._forward_micro_batch = fake_forward_micro_batch
    actor._optimizer_step = lambda: torch.tensor(0.0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: "cpu")
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_policy_loss_fn", lambda _: fake_policy_loss)

    tensors = TensorDict(
        {
            "responses": torch.tensor([[1, 2]]),
            "response_mask": torch.ones(1, 2),
            "input_ids": torch.tensor([[0, 1, 2]]),
            "attention_mask": torch.ones(1, 3),
            "position_ids": torch.arange(3).unsqueeze(0),
            "old_log_probs": torch.zeros(1, 2),
            "advantages": torch.tensor([[-2.0, -1.0]]),
        },
        batch_size=[1],
    )
    data = DataProto(
        batch=tensors,
        meta_info={
            "temperature": 1.0,
            "eos_mode": "semantic_class",
            "semantic_eos_token_ids": [1, 3],
        },
    )

    actor.update_policy(data)

    torch.testing.assert_close(captured["advantages"], torch.tensor([[-2.0, -1.0]]))


def test_forward_micro_batch_returns_fixed_token_log_probs_for_response_positions():
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = FixedLogitsModel(logits)
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(entropy_checkpointing=False)

    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": torch.tensor([[2, 3]]),
    }

    _, _, _, _, tracked_log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        top_k=2,
        tracked_token_ids=[4, 1],
    )

    expected = torch.log_softmax(logits[:, 1:3, :], dim=-1)[..., [4, 1]]
    torch.testing.assert_close(tracked_log_probs, expected)


def test_tracking_fixed_tokens_does_not_change_existing_forward_outputs():
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = FixedLogitsModel(logits)
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(entropy_checkpointing=False)
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": torch.tensor([[2, 3]]),
    }

    baseline_outputs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        top_k=2,
    )
    tracked_outputs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        top_k=2,
        tracked_token_ids=[4, 1],
    )

    assert baseline_outputs[0] is tracked_outputs[0] is None
    for baseline, tracked in zip(baseline_outputs[1:4], tracked_outputs[1:4], strict=True):
        torch.testing.assert_close(tracked, baseline)


def test_forward_micro_batch_tracks_fixed_tokens_when_top_k_is_disabled(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.actor.dp_actor.logprobs_from_logits",
        lambda values, labels: torch.log_softmax(values, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1),
    )
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = FixedLogitsModel(logits)
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(entropy_checkpointing=False)

    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": torch.tensor([[2, 3]]),
    }

    _, sampled_log_probs, topk_ids, topk_log_probs, tracked_log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=False,
        top_k=0,
        tracked_token_ids=[4, 1],
    )

    response_logits = logits[:, 1:3, :]
    expected_sampled = torch.log_softmax(response_logits, dim=-1).gather(
        dim=-1, index=micro_batch["responses"].unsqueeze(-1)
    ).squeeze(-1)
    expected_tracked = torch.log_softmax(response_logits, dim=-1)[..., [4, 1]]

    torch.testing.assert_close(sampled_log_probs, expected_sampled)
    torch.testing.assert_close(tracked_log_probs, expected_tracked)
    assert topk_ids is None
    assert topk_log_probs is None


@pytest.mark.parametrize("mode", ["semantic_class", "canonical"])
def test_forward_micro_batch_applies_opt_in_student_eos_policy(mode, monkeypatch):
    monkeypatch.setattr(
        "verl.workers.actor.dp_actor.logprobs_from_logits",
        lambda values, labels: torch.log_softmax(values, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1),
    )
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = FixedLogitsModel(logits)
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(entropy_checkpointing=False)
    responses = torch.tensor([[2, 3]])
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": responses,
    }

    _, actual, *_ = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        eos_mode=mode,
        semantic_eos_token_ids=[2, 4],
    )

    response_logits = logits[:, 1:3, :]
    if mode == "semantic_class":
        full_log_probs = torch.log_softmax(response_logits, dim=-1)
        expected = full_log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
        expected[:, 0] = torch.logsumexp(full_log_probs[:, 0, [2, 4]], dim=-1)
    else:
        kept_logits = response_logits.clone()
        kept_logits[..., 4] = -torch.inf
        expected = torch.log_softmax(kept_logits, dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)


def test_forward_micro_batch_semantic_class_aggregates_three_terminal_tokens(monkeypatch):
    monkeypatch.setattr(
        "verl.workers.actor.dp_actor.logprobs_from_logits",
        lambda values, labels: torch.log_softmax(values, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1),
    )
    logits = torch.arange(1 * 4 * 6, dtype=torch.float32).reshape(1, 4, 6)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = FixedLogitsModel(logits)
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.device_name = "cpu"
    actor.config = SimpleNamespace(entropy_checkpointing=False)
    responses = torch.tensor([[2, 3]])
    micro_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0),
        "responses": responses,
    }

    _, actual, *_ = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        eos_mode="semantic_class",
        semantic_eos_token_ids=[2, 4, 5],
    )

    full_log_probs = torch.log_softmax(logits[:, 1:3, :], dim=-1)
    expected = full_log_probs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    expected[:, 0] = torch.logsumexp(full_log_probs[:, 0, [2, 4, 5]], dim=-1)
    torch.testing.assert_close(actual, expected)
