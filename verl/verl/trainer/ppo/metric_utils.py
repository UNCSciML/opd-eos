# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Metrics related to the PPO trainer.
"""

import csv
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_tracked_token_probability_metrics(
    tracked_token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    token_names: list[str],
) -> dict[str, float]:
    """Aggregate fixed-token probabilities without affecting training tensors."""
    if tracked_token_log_probs.size(-1) != len(token_names):
        raise ValueError("token_names must match the tracked-token dimension")

    response_mask = response_mask.to(device=tracked_token_log_probs.device, dtype=torch.bool)
    valid_sequences = response_mask.any(dim=-1)
    probabilities = tracked_token_log_probs.detach().float().exp()

    token_counts = response_mask.sum(dim=-1).clamp_min(1).unsqueeze(-1)
    sequence_means = (probabilities * response_mask.unsqueeze(-1)).sum(dim=1) / token_counts

    positions = torch.arange(response_mask.size(1), device=response_mask.device).expand_as(response_mask)
    last_positions = positions.masked_fill(~response_mask, -1).max(dim=-1).values.clamp_min(0)
    last_probabilities = probabilities.gather(
        dim=1,
        index=last_positions[:, None, None].expand(-1, 1, probabilities.size(-1)),
    ).squeeze(1)

    metrics = {}
    for index, token_name in enumerate(token_names):
        if valid_sequences.any():
            sequence_mean = sequence_means[valid_sequences, index].mean().item()
            last_token = last_probabilities[valid_sequences, index].mean().item()
        else:
            sequence_mean = 0.0
            last_token = 0.0
        metrics[f"student_eos_prob/{token_name}/sequence_mean"] = sequence_mean
        metrics[f"student_eos_prob/{token_name}/last_token"] = last_token
    return metrics


def extract_teacher_eos_at_student_eos(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_eos_log_probs: torch.Tensor,
    student_eos_token_id: int | None = None,
    terminal_token_ids: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Extract detached teacher terminal probabilities where a response ends in a terminal token."""
    if teacher_eos_log_probs.ndim != 3 or teacher_eos_log_probs.size(-1) < 1:
        raise ValueError("teacher_eos_log_probs must have shape [batch, response_length, K] with K >= 1")
    if teacher_eos_log_probs.shape[:2] != responses.shape or response_mask.shape != responses.shape:
        raise ValueError("responses, response_mask, and teacher EOS positions must have matching shapes")
    if terminal_token_ids is None:
        if student_eos_token_id is None:
            raise ValueError("terminal_token_ids or student_eos_token_id is required")
        terminal_token_ids = [int(student_eos_token_id)]
    terminal_token_ids = [int(token_id) for token_id in terminal_token_ids]
    if not terminal_token_ids:
        raise ValueError("terminal_token_ids must not be empty")

    mask = response_mask.to(device=responses.device, dtype=torch.bool)
    lengths = mask.sum(dim=-1)
    positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
    last_positions = positions.masked_fill(~mask, -1).max(dim=-1).values.clamp_min(0)
    last_tokens = responses.gather(dim=-1, index=last_positions.unsqueeze(-1)).squeeze(-1)
    terminal_ids = torch.as_tensor(terminal_token_ids, dtype=responses.dtype, device=responses.device)
    selected_sequences = (lengths > 0) & (last_tokens.unsqueeze(-1) == terminal_ids).any(dim=-1)

    num_eos_tokens = teacher_eos_log_probs.size(-1)
    selected_log_probs = teacher_eos_log_probs.gather(
        dim=1,
        index=last_positions[:, None, None].expand(-1, 1, num_eos_tokens),
    ).squeeze(1)[selected_sequences].detach().float()
    selected_probs = selected_log_probs.exp()
    eos_mass = selected_probs.sum(dim=-1)
    eos_shares = torch.softmax(selected_log_probs, dim=-1)

    def as_numpy(values: torch.Tensor) -> np.ndarray:
        return values.cpu().numpy()

    records = {
        "response_length": as_numpy(lengths[selected_sequences].detach().to(dtype=torch.int64)),
        "sampled_terminal_id": as_numpy(last_tokens[selected_sequences].detach().to(dtype=torch.int64)),
        "teacher_eos_probs": as_numpy(selected_probs),
        "teacher_eos_shares": as_numpy(eos_shares),
        "teacher_eos_mass": as_numpy(eos_mass),
    }
    if num_eos_tokens == 2:
        records.update(
            {
                "teacher_prob_e1": as_numpy(selected_probs[:, 0]),
                "teacher_prob_e2": as_numpy(selected_probs[:, 1]),
                "teacher_e1_share": as_numpy(eos_shares[:, 0]),
                "teacher_e2_share": as_numpy(eos_shares[:, 1]),
                "logprob_e2_minus_e1": as_numpy(selected_log_probs[:, 1] - selected_log_probs[:, 0]),
            }
        )
    return records


def save_teacher_eos_diagnostic_npz(
    *,
    output_dir: str | Path,
    global_step: int,
    eos_token_ids: list[int],
    records: dict[str, np.ndarray],
) -> Path:
    """Persist one diagnostic step without storing prompts, responses, or full logits."""
    if not eos_token_ids:
        raise ValueError("teacher EOS diagnostics require at least one EOS token id")
    if "teacher_eos_probs" in records and np.asarray(records["teacher_eos_probs"]).shape[-1] != len(eos_token_ids):
        raise ValueError("teacher_eos_probs final dimension must match eos_token_ids")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"step_{int(global_step):04d}_teacher_eos_at_student_eos.npz"
    np.savez_compressed(
        output_path,
        global_step=np.asarray(int(global_step), dtype=np.int64),
        eos_token_ids=np.asarray(eos_token_ids, dtype=np.int64),
        **records,
    )

    count = int(len(records["response_length"]))

    def array_mean(name: str) -> float:
        values = np.asarray(records[name])
        return float(values.mean()) if values.size else float("nan")

    summary_row: dict[str, float | int] = {
        "global_step": int(global_step),
        "count": count,
        "teacher_eos_mass_mean": array_mean("teacher_eos_mass"),
        "response_length_mean": array_mean("response_length"),
    }
    if "teacher_eos_probs" in records:
        probabilities = np.asarray(records["teacher_eos_probs"])
        shares = np.asarray(records["teacher_eos_shares"])
        for index, token_id in enumerate(eos_token_ids):
            summary_row[f"teacher_prob_token_{token_id}_mean"] = (
                float(probabilities[:, index].mean()) if probabilities.size else float("nan")
            )
            summary_row[f"teacher_share_token_{token_id}_mean"] = (
                float(shares[:, index].mean()) if shares.size else float("nan")
            )
    if len(eos_token_ids) == 2 and "teacher_prob_e1" in records:
        summary_row.update(
            {
                "teacher_prob_e1_mean": array_mean("teacher_prob_e1"),
                "teacher_prob_e2_mean": array_mean("teacher_prob_e2"),
                "teacher_e1_share_mean": array_mean("teacher_e1_share"),
                "teacher_e2_share_mean": array_mean("teacher_e2_share"),
                "logprob_e2_minus_e1_mean": array_mean("logprob_e2_minus_e1"),
            }
        )
    summary_path = output_dir / "summary.csv"
    fieldnames = list(summary_row)
    existing_rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = [
                row for row in csv.DictReader(handle) if int(row["global_step"]) != int(global_step)
            ]
    existing_rows.append({key: str(value) for key, value in summary_row.items()})
    existing_rows.sort(key=lambda row: int(row["global_step"]))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
    return output_path


def attach_eos_semantics_metadata(batch: Any, rollout_config: Any) -> tuple[str, list[int], float]:
    """Attach EOS diagnostic settings before any optional reward-model branch."""
    eos_mode = rollout_config.get("eos_mode", "baseline")
    semantic_eos_token_ids = list(rollout_config.get("semantic_eos_token_ids", []))
    eos_mapping_epsilon = float(rollout_config.get("eos_mapping_epsilon", 1e-12))
    batch.meta_info["eos_mode"] = eos_mode
    batch.meta_info["semantic_eos_token_ids"] = semantic_eos_token_ids
    batch.meta_info["eos_mapping_epsilon"] = eos_mapping_epsilon
    return eos_mode, semantic_eos_token_ids, eos_mapping_epsilon


def _eos_probability_metrics(
    *,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    token_names: list[str],
    prefix: str,
) -> dict[str, float]:
    probabilities = log_probs.detach().float().exp()
    mask = response_mask.to(device=probabilities.device, dtype=torch.bool)
    valid_sequences = mask.any(dim=-1)
    token_counts = mask.sum(dim=-1).clamp_min(1).unsqueeze(-1)
    sequence_means = (probabilities * mask.unsqueeze(-1)).sum(dim=1) / token_counts
    positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
    last_positions = positions.masked_fill(~mask, -1).max(dim=-1).values.clamp_min(0)
    last_probabilities = probabilities.gather(
        dim=1,
        index=last_positions[:, None, None].expand(-1, 1, probabilities.size(-1)),
    ).squeeze(1)

    metrics = {}
    labels_and_indices = [(name, index) for index, name in enumerate(token_names)]
    labels_and_indices.append(("sum", None))
    for name, index in labels_and_indices:
        per_sequence = sequence_means.sum(dim=-1) if index is None else sequence_means[:, index]
        at_last = last_probabilities.sum(dim=-1) if index is None else last_probabilities[:, index]
        metrics[f"{prefix}/{name}/sequence_mean"] = (
            per_sequence[valid_sequences].mean().item() if valid_sequences.any() else 0.0
        )
        metrics[f"{prefix}/{name}/last_token"] = (
            at_last[valid_sequences].mean().item() if valid_sequences.any() else 0.0
        )
    return metrics


def compute_eos_diagnostic_metrics(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    student_eos_log_probs: torch.Tensor,
    teacher_eos_log_probs: torch.Tensor,
    eos_token_ids: list[int],
    token_names: list[str],
) -> dict[str, float]:
    """Compute detached EOS diagnostics for W&B without changing optimization tensors."""
    if not eos_token_ids or len(token_names) != len(eos_token_ids):
        raise ValueError("EOS diagnostics require one name per non-empty EOS token-id list")
    if (
        student_eos_log_probs.size(-1) != len(eos_token_ids)
        or teacher_eos_log_probs.size(-1) != len(eos_token_ids)
    ):
        raise ValueError("EOS diagnostic log-prob dimensions must match eos_token_ids")

    mask = response_mask.to(device=responses.device, dtype=torch.bool)
    lengths = mask.sum(dim=-1)
    valid_sequences = lengths > 0
    valid_lengths = lengths[valid_sequences].float()
    eos_token_ids = [int(token_id) for token_id in eos_token_ids]
    positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
    last_positions = positions.masked_fill(~mask, -1).max(dim=-1).values.clamp_min(0)
    last_tokens = responses.gather(dim=-1, index=last_positions.unsqueeze(-1)).squeeze(-1)

    metrics: dict[str, float] = {
        "response_length/mean": valid_lengths.mean().item() if valid_lengths.numel() else 0.0,
        "response_length/min": valid_lengths.min().item() if valid_lengths.numel() else 0.0,
        "response_length/max": valid_lengths.max().item() if valid_lengths.numel() else 0.0,
        "response_length/median": torch.quantile(valid_lengths, 0.5).item() if valid_lengths.numel() else 0.0,
        "response_length/p95": torch.quantile(valid_lengths, 0.95).item() if valid_lengths.numel() else 0.0,
    }
    denominator = valid_sequences.sum().clamp_min(1).float()
    finish_masks = [valid_sequences & (last_tokens == token_id) for token_id in eos_token_ids]
    finishes_terminal = torch.stack(finish_masks, dim=-1).any(dim=-1)
    finishes_max = valid_sequences & (lengths == mask.size(1)) & ~finishes_terminal
    finishes_other = valid_sequences & ~finishes_terminal & ~finishes_max
    metrics.update(
        {
            "finish_reason/semantic_terminal_fraction": (finishes_terminal.sum() / denominator).item(),
            "finish_reason/max_tokens_fraction": (finishes_max.sum() / denominator).item(),
            "finish_reason/other_fraction": (finishes_other.sum() / denominator).item(),
        }
    )
    for token_id, finish_mask in zip(eos_token_ids, finish_masks, strict=True):
        metrics[f"finish_reason/token_{token_id}_fraction"] = (finish_mask.sum() / denominator).item()
    if len(eos_token_ids) == 2:
        metrics["finish_reason/e1_fraction"] = (finish_masks[0].sum() / denominator).item()
        metrics["finish_reason/e2_fraction"] = (finish_masks[1].sum() / denominator).item()

    advantage_values = advantages.detach()
    if advantage_values.dim() == 3:
        advantage_values = advantage_values.sum(dim=-1)
    for token_id, token_name in zip(eos_token_ids, token_names, strict=True):
        sampled_mask = mask & (responses == token_id)
        sampled_advantages = advantage_values[sampled_mask]
        sampled_count = float(sampled_mask.sum().item())
        sampled_mean_advantage = (
            sampled_advantages.float().mean().item() if sampled_advantages.numel() else 0.0
        )
        metrics[f"sampled_eos/{token_name}/count"] = sampled_count
        metrics[f"sampled_eos/{token_name}/mean_advantage"] = sampled_mean_advantage
        metrics[f"sampled_eos/token_{token_id}/count"] = sampled_count
        metrics[f"sampled_eos/token_{token_id}/mean_advantage"] = sampled_mean_advantage

    if len(eos_token_ids) == 2:
        e2_positions = mask & (responses == eos_token_ids[1])
        sequences_with_e2 = e2_positions.any(dim=-1)
        first_e2 = positions.masked_fill(~e2_positions, mask.size(1)).min(dim=-1).values
        tokens_after_e2 = (lengths - first_e2 - 1).clamp_min(0).float()
        metrics["eos/e2/tokens_after_mean"] = (
            tokens_after_e2[sequences_with_e2].mean().item() if sequences_with_e2.any() else 0.0
        )
        metrics["eos/e2/sequence_fraction"] = (sequences_with_e2.sum() / denominator).item()

    metrics.update(
        _eos_probability_metrics(
            log_probs=student_eos_log_probs,
            response_mask=mask,
            token_names=token_names,
            prefix="student_eos_prob",
        )
    )
    metrics.update(
        _eos_probability_metrics(
            log_probs=teacher_eos_log_probs,
            response_mask=mask,
            token_names=token_names,
            prefix="teacher_eos_prob",
        )
    )
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    sequence_true_reward = batch.batch["true_reward_score"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]
    non_aborted_sequence_true_reward = sequence_true_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    true_reward_mean = torch.mean(non_aborted_sequence_true_reward).detach().item()
    true_reward_max = torch.max(non_aborted_sequence_true_reward).detach().item()
    true_reward_min = torch.min(non_aborted_sequence_true_reward).detach().item()


    # Handle 2D and 3D advantages/returns with different strategies:
    # - 2D: normal computation on all values (masked by response_mask)
    # - 3D max/min: computed on entire 3D tensor (including zeros)
    # - 3D mean: computed only on non-zero values
    
    # For 2D, use response_mask to filter
    if advantages.dim() == 2:
        valid_adv_all = torch.masked_select(advantages, response_mask)  # for max/min
        valid_adv_nonzero = valid_adv_all[valid_adv_all != 0]  # for mean
        advantages_2d = advantages
    else:
        # 3D case: (batch, seq, k)
        # All values for max/min (including zeros)
        valid_adv_all = advantages.view(-1)
        # Only non-zero values for mean
        valid_adv_nonzero = advantages[advantages != 0]
        # Keep 2D version for position-based metrics
        valid_k_mask = (advantages.abs() > 1e-9).float()
        valid_k_count = valid_k_mask.sum(dim=-1).clamp(min=1.0)
        advantages_2d = (advantages * valid_k_mask).sum(dim=-1) / valid_k_count
        
    if returns.dim() == 2:
        valid_returns_all = torch.masked_select(returns, response_mask)
        valid_returns_nonzero = valid_returns_all[valid_returns_all != 0]
        returns_2d = returns
    else:
        # 3D case
        valid_returns_all = returns.view(-1)
        valid_returns_nonzero = returns[returns != 0]
        valid_k_mask_ret = (returns.abs() > 1e-9).float()
        valid_k_count_ret = valid_k_mask_ret.sum(dim=-1).clamp(min=1.0)
        returns_2d = (returns * valid_k_mask_ret).sum(dim=-1) / valid_k_count_ret

    partial_adv_metrics: dict[str, Any] = {}
    
    # For 3D case: compute two mean values
    # 1. Mean after summing over K (matches loss calculation): sum then mean over (B, T)
    # 2. Mean over all non-zero values (overall mean)
    if advantages.dim() == 3:
        # Method 1: sum over K, then mean over valid positions
        adv_sum_over_k = advantages.sum(dim=-1)  # (B, T, K) -> (B, T)
        valid_adv_sum = torch.masked_select(adv_sum_over_k, response_mask)
        partial_adv_metrics["critic/advantages/mean_sum_over_k"] = (
            torch.mean(valid_adv_sum).detach().item() if valid_adv_sum.numel() > 0 else 0.0
        )
        
        # Method 2: mean over all non-zero values (flattened)
        partial_adv_metrics["critic/advantages/mean_overall"] = (
            torch.mean(valid_adv_nonzero).detach().item() if valid_adv_nonzero.numel() > 0 else 0.0
        )
    else:
        # 2D case: single mean value
        partial_adv_metrics["critic/advantages/mean"] = (
            torch.mean(valid_adv_nonzero).detach().item() if valid_adv_nonzero.numel() > 0 else 0.0
        )
    
    # Debug: Add more detailed advantage statistics to diagnose gradient issues
    # Use valid_adv_all for max/min (includes zeros in 3D case), valid_adv_nonzero for mean/std
    if valid_adv_nonzero.numel() > 0:
        partial_adv_metrics["critic/advantages/std"] = torch.std(valid_adv_nonzero).detach().item()
        partial_adv_metrics["critic/advantages/abs_mean"] = torch.mean(torch.abs(valid_adv_nonzero)).detach().item()
        # Count extreme values (|adv| > 5) among non-zero values
        extreme_count = (torch.abs(valid_adv_nonzero) > 5.0).sum().item()
        partial_adv_metrics["critic/advantages/extreme_count"] = extreme_count
    
    # max/min computed on all values (including zeros for 3D)
    if valid_adv_all.numel() > 0:
        adv_abs_max = torch.max(torch.abs(valid_adv_all)).detach().item()
        partial_adv_metrics["critic/advantages/abs_max"] = adv_abs_max

    if "token_level_advantage_direct" in batch.batch.keys():
        direct_adv = batch.batch["token_level_advantage_direct"]
        # Handle 3D advantages (top-k case) by summing over k dimension (matches loss)
        if direct_adv.dim() == 3:
            direct_adv = direct_adv.sum(dim=-1)
        valid_direct_adv = torch.masked_select(direct_adv, response_mask)
        partial_adv_metrics["critic/advantages/direct_mean"] = (
            torch.mean(valid_direct_adv).detach().item() if valid_direct_adv.numel() > 0 else 0.0
        )
        partial_adv_metrics["critic/advantages/direct_max"] = (
            torch.max(valid_direct_adv).detach().item() if valid_direct_adv.numel() > 0 else 0.0
        )
        partial_adv_metrics["critic/advantages/direct_min"] = (
            torch.min(valid_direct_adv).detach().item() if valid_direct_adv.numel() > 0 else 0.0
        )


    max_response_length = response_mask.size(-1)
    step = 2 * 1024
    if max_response_length >= step:
        token_positions = torch.arange(max_response_length, device=response_mask.device)[None, :]
        for cutoff in range(step, max_response_length + 1, step):
            cutoff_mask = response_mask & (token_positions < cutoff)
            # Use 2D advantages for position-based cutoff metrics
            cutoff_adv = torch.masked_select(advantages_2d, cutoff_mask)
            partial_adv_metrics[f"critic/advantages/mean_first_{cutoff // 1024}k"] = (
                torch.mean(cutoff_adv).detach().item() if cutoff_adv.numel() > 0 else 0.0
            )
            
            if "token_level_advantage_direct" in batch.batch.keys():
                cutoff_direct_adv = torch.masked_select(direct_adv, cutoff_mask)
                partial_adv_metrics[f"critic/advantages/direct_mean_first_{cutoff // 1024}k"] = (
                    torch.mean(cutoff_direct_adv).detach().item() if cutoff_direct_adv.numel() > 0 else 0.0
                )

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        # Use 2D returns for variance calculation to match values shape
        valid_returns_2d = torch.masked_select(returns_2d, response_mask)
        if valid_returns_2d.numel() > 0:
            return_diff_var = torch.var(valid_returns_2d - valid_values)
            return_var = torch.var(valid_returns_2d)
        else:
            return_diff_var = torch.tensor(0.0)
            return_var = torch.tensor(0.0)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # true reward
        "critic/true_reward/mean": true_reward_mean,
        "critic/true_reward/max": true_reward_max,
        "critic/true_reward/min": true_reward_min,
        # adv: max/min on all values; two mean values for 3D case
        "critic/advantages/max": torch.max(valid_adv_all).detach().item() if valid_adv_all.numel() > 0 else 0.0,
        "critic/advantages/min": torch.min(valid_adv_all).detach().item() if valid_adv_all.numel() > 0 else 0.0,
        **partial_adv_metrics,
        # returns: max/min on all values; two mean values for 3D case
        "critic/returns/max": torch.max(valid_returns_all).detach().item() if valid_returns_all.numel() > 0 else 0.0,
        "critic/returns/min": torch.min(valid_returns_all).detach().item() if valid_returns_all.numel() > 0 else 0.0,
        "critic/returns/mean": (
            torch.mean(valid_returns_nonzero).detach().item() if valid_returns_nonzero.numel() > 0 else 0.0
        ),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
                "critic/values/max": torch.max(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
                "critic/values/min": torch.min(valid_values).detach().item() if valid_values.numel() > 0 else 0.0,
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
