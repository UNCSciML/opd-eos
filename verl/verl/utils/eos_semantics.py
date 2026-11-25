"""Tensor transforms for optional OPD EOS semantics experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

SUPPORTED_EOS_MODES = frozenset({"baseline", "two_stop", "teacher_map", "semantic_class", "canonical"})


def validate_eos_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_EOS_MODES:
        choices = ", ".join(sorted(SUPPORTED_EOS_MODES))
        raise ValueError(f"Unsupported EOS mode {mode!r}; expected one of: {choices}")
    return normalized


def gather_fixed_token_log_probs(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Gather fixed-token log-probs without materializing full-vocabulary log-probs."""
    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=logits.device)
    selected_logits = logits.index_select(dim=-1, index=ids)
    return selected_logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def validate_semantic_eos_token_ids(token_ids: Sequence[int], *, mode: str) -> list[int]:
    ids = [int(token_id) for token_id in token_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Semantic EOS token ids must be a non-empty list of distinct ids")
    if mode in {"teacher_map", "canonical"} and len(ids) < 2:
        raise ValueError(f"{mode} requires at least two distinct semantic EOS token ids")
    return ids


def _is_any_token_id(values: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    ids = torch.as_tensor(list(token_ids), dtype=values.dtype, device=values.device)
    return (values.unsqueeze(-1) == ids).any(dim=-1)


def map_teacher_sampled_log_probs(
    *,
    raw_sampled_log_probs: torch.Tensor,
    teacher_eos_log_probs: torch.Tensor,
    sampled_ids: torch.Tensor,
    mode: str,
    semantic_eos_token_ids: Sequence[int],
    epsilon: float,
) -> torch.Tensor:
    """Apply the selected teacher-side EOS interpretation to sampled actions."""
    mode = validate_eos_mode(mode)
    if mode in {"baseline", "two_stop"}:
        return raw_sampled_log_probs

    eos_ids = validate_semantic_eos_token_ids(semantic_eos_token_ids, mode=mode)
    if teacher_eos_log_probs.size(-1) != len(eos_ids):
        raise ValueError(
            "teacher_eos_log_probs final dimension must match semantic_eos_token_ids "
            f"({teacher_eos_log_probs.size(-1)} != {len(eos_ids)})"
        )

    eos_sample = _is_any_token_id(sampled_ids, eos_ids)
    combined_log_prob = torch.logsumexp(teacher_eos_log_probs, dim=-1)
    if mode == "semantic_class":
        return torch.where(eos_sample, combined_log_prob, raw_sampled_log_probs)

    if epsilon <= 0:
        raise ValueError("EOS mapping epsilon must be positive")
    canonical_id, *other_ids = eos_ids
    log_other_count = math.log(len(other_ids))
    log_epsilon = math.log(epsilon)
    # Use epsilon for each other EOS, capped so canonical retains at least half of the total.
    other_log_prob = torch.clamp(combined_log_prob - (math.log(2) + log_other_count), max=log_epsilon)
    other_fraction = torch.clamp(log_epsilon + log_other_count - combined_log_prob, max=-math.log(2)).exp()
    canonical_log_prob = combined_log_prob + torch.log1p(-other_fraction)

    mapped_eos_log_prob = torch.where(sampled_ids == canonical_id, canonical_log_prob, other_log_prob)
    return torch.where(eos_sample, mapped_eos_log_prob, raw_sampled_log_probs)


def student_policy_log_probs(
    *,
    logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    raw_sampled_log_probs: torch.Tensor,
    mode: str,
    semantic_eos_token_ids: Sequence[int],
) -> torch.Tensor:
    """Return sampled-action log-probs under the selected student action space."""
    mode = validate_eos_mode(mode)
    if mode in {"baseline", "two_stop"}:
        return raw_sampled_log_probs

    eos_ids = validate_semantic_eos_token_ids(semantic_eos_token_ids, mode=mode)
    if mode == "teacher_map":
        return raw_sampled_log_probs

    if mode == "semantic_class":
        eos_log_probs = gather_fixed_token_log_probs(logits, eos_ids)
        combined_log_prob = torch.logsumexp(eos_log_probs, dim=-1)
        eos_sample = _is_any_token_id(sampled_ids, eos_ids)
        return torch.where(eos_sample, combined_log_prob, raw_sampled_log_probs)

    blocked_ids = eos_ids[1:]
    if torch.any(_is_any_token_id(sampled_ids, blocked_ids)):
        raise ValueError(f"Canonical EOS policy sampled a blocked EOS token from {blocked_ids}")

    # Reduce contiguous allowed slices so we avoid both cancellation and copying the full logits.
    log_normalizer = None
    start = 0
    for end in [*sorted(blocked_ids), logits.size(-1)]:
        if start < end:
            chunk_normalizer = torch.logsumexp(logits[..., start:end], dim=-1)
            log_normalizer = (
                chunk_normalizer if log_normalizer is None else torch.logaddexp(log_normalizer, chunk_normalizer)
            )
        start = end + 1
    sampled_logits = logits.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
    return sampled_logits - log_normalizer
