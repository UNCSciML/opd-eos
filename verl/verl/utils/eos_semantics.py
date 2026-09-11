"""Tensor transforms for optional OPD EOS semantics experiments."""

from __future__ import annotations

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


def _validate_semantic_eos_token_ids(token_ids: Sequence[int], *, exactly_two: bool = False) -> list[int]:
    ids = [int(token_id) for token_id in token_ids]
    if exactly_two and (len(ids) != 2 or ids[0] == ids[1]):
        raise ValueError("This EOS mode requires exactly two distinct semantic EOS token ids")
    if not exactly_two and (not ids or len(set(ids)) != len(ids)):
        raise ValueError("Semantic EOS token ids must be a non-empty list of distinct ids")
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

    eos_ids = _validate_semantic_eos_token_ids(
        semantic_eos_token_ids,
        exactly_two=mode in {"teacher_map", "canonical"},
    )
    if teacher_eos_log_probs.size(-1) != len(eos_ids):
        raise ValueError(
            "teacher_eos_log_probs final dimension must match semantic_eos_token_ids "
            f"({teacher_eos_log_probs.size(-1)} != {len(eos_ids)})"
        )

    eos_sample = _is_any_token_id(sampled_ids, eos_ids)
    combined_log_prob = torch.logsumexp(teacher_eos_log_probs, dim=-1)
    if mode == "semantic_class":
        return torch.where(eos_sample, combined_log_prob, raw_sampled_log_probs)

    e1, e2 = eos_ids
    if epsilon <= 0:
        raise ValueError("EOS mapping epsilon must be positive")
    combined_prob = combined_log_prob.exp()
    requested_epsilon = torch.full_like(combined_prob, epsilon)
    mapped_e2_prob = torch.minimum(requested_epsilon, combined_prob * 0.5)
    mapped_e1_log_prob = (combined_prob - mapped_e2_prob).log()
    mapped_e2_log_prob = mapped_e2_prob.log()

    mapped = torch.where(sampled_ids == e1, mapped_e1_log_prob, raw_sampled_log_probs)
    return torch.where(sampled_ids == e2, mapped_e2_log_prob, mapped)


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

    eos_ids = _validate_semantic_eos_token_ids(
        semantic_eos_token_ids,
        exactly_two=mode in {"teacher_map", "canonical"},
    )
    if mode == "teacher_map":
        return raw_sampled_log_probs

    if mode == "semantic_class":
        eos_log_probs = gather_fixed_token_log_probs(logits, eos_ids)
        combined_log_prob = torch.logsumexp(eos_log_probs, dim=-1)
        eos_sample = _is_any_token_id(sampled_ids, eos_ids)
        return torch.where(eos_sample, combined_log_prob, raw_sampled_log_probs)

    _, e2 = eos_ids
    if torch.any(sampled_ids == e2):
        raise ValueError(f"Canonical EOS policy sampled blocked EOS token {e2}")

    log_normalizer = torch.logsumexp(logits, dim=-1)
    blocked_logit = logits[..., e2]
    blocked_fraction = (blocked_logit - log_normalizer).exp()
    max_fraction = 1.0 - torch.finfo(blocked_fraction.dtype).eps
    log_normalizer_without_e2 = log_normalizer + torch.log1p(-blocked_fraction.clamp_max(max_fraction))
    sampled_logits = logits.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
    return sampled_logits - log_normalizer_without_e2
