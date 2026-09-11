import math

import pytest
import torch

from verl.utils.eos_semantics import (
    gather_fixed_token_log_probs,
    map_teacher_sampled_log_probs,
    student_policy_log_probs,
    validate_eos_mode,
)


def test_validate_eos_mode_accepts_the_experiment_matrix_and_rejects_unknown_values():
    for mode in ("baseline", "two_stop", "teacher_map", "semantic_class", "canonical"):
        assert validate_eos_mode(mode) == mode

    with pytest.raises(ValueError, match="Unsupported EOS mode"):
        validate_eos_mode("typo")


def test_baseline_and_two_stop_leave_sampled_log_probs_bitwise_unchanged():
    raw = torch.tensor([[-1.25, -0.75]], dtype=torch.float32)
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]]])
    sampled_ids = torch.tensor([[1, 3]])

    for mode in ("baseline", "two_stop", "teacher_map"):
        output = student_policy_log_probs(
            logits=logits,
            sampled_ids=sampled_ids,
            raw_sampled_log_probs=raw,
            mode=mode,
            semantic_eos_token_ids=[1, 3],
        )
        assert output is raw


def test_teacher_mapping_conserves_eos_mass_and_keeps_non_eos_sample_unchanged():
    sampled_ids = torch.tensor([[1, 3, 2]])
    raw = torch.log(torch.tensor([[0.20, 0.40, 0.50]], dtype=torch.float64))
    teacher_eos = torch.log(
        torch.tensor([[[0.20, 0.30], [0.10, 0.40], [0.25, 0.15]]], dtype=torch.float64)
    )

    mapped = map_teacher_sampled_log_probs(
        raw_sampled_log_probs=raw,
        teacher_eos_log_probs=teacher_eos,
        sampled_ids=sampled_ids,
        mode="teacher_map",
        semantic_eos_token_ids=[1, 3],
        epsilon=0.01,
    )

    torch.testing.assert_close(mapped.exp(), torch.tensor([[0.49, 0.01, 0.50]], dtype=torch.float64))
    torch.testing.assert_close(mapped.exp()[0, :2].sum(), torch.tensor(0.50, dtype=torch.float64))


def test_semantic_class_uses_combined_eos_probability_for_both_surface_tokens():
    sampled_ids = torch.tensor([[1, 3, 2]])
    raw = torch.log(torch.tensor([[0.20, 0.40, 0.50]], dtype=torch.float64))
    eos_log_probs = torch.log(
        torch.tensor([[[0.20, 0.30], [0.10, 0.40], [0.25, 0.15]]], dtype=torch.float64)
    )

    teacher = map_teacher_sampled_log_probs(
        raw_sampled_log_probs=raw,
        teacher_eos_log_probs=eos_log_probs,
        sampled_ids=sampled_ids,
        mode="semantic_class",
        semantic_eos_token_ids=[1, 3],
        epsilon=1e-12,
    )
    torch.testing.assert_close(teacher.exp(), torch.tensor([[0.50, 0.50, 0.50]], dtype=torch.float64))

    logits = eos_log_probs.new_tensor(
        [[[math.log(0.1), math.log(0.2), math.log(0.4), math.log(0.3)]] * 3]
    )
    student = student_policy_log_probs(
        logits=logits,
        sampled_ids=sampled_ids,
        raw_sampled_log_probs=raw,
        mode="semantic_class",
        semantic_eos_token_ids=[1, 3],
    )
    torch.testing.assert_close(student.exp(), torch.tensor([[0.50, 0.50, 0.50]], dtype=torch.float64))


def test_canonical_policy_masks_e2_and_renormalizes_every_remaining_action():
    logits = torch.log(torch.tensor([[[0.20, 0.30, 0.10, 0.40]] * 3], dtype=torch.float64))
    sampled_ids = torch.tensor([[0, 1, 2]])
    raw = torch.log(torch.tensor([[0.20, 0.30, 0.10]], dtype=torch.float64))

    canonical = student_policy_log_probs(
        logits=logits,
        sampled_ids=sampled_ids,
        raw_sampled_log_probs=raw,
        mode="canonical",
        semantic_eos_token_ids=[1, 3],
    )
    torch.testing.assert_close(
        canonical.exp(),
        torch.tensor([[0.20 / 0.60, 0.30 / 0.60, 0.10 / 0.60]], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="sampled blocked EOS token"):
        student_policy_log_probs(
            logits=logits[:, :1],
            sampled_ids=torch.tensor([[3]]),
            raw_sampled_log_probs=raw[:, :1],
            mode="canonical",
            semantic_eos_token_ids=[1, 3],
        )


def test_semantic_class_aggregates_three_terminal_tokens_and_leaves_other_actions_unchanged():
    sampled_ids = torch.tensor([[1, 3, 4, 2]])
    raw = torch.log(torch.tensor([[0.20, 0.30, 0.10, 0.55]], dtype=torch.float64))
    teacher_eos = torch.log(
        torch.tensor(
            [[[0.10, 0.20, 0.30], [0.05, 0.15, 0.20], [0.20, 0.10, 0.10], [0.12, 0.08, 0.05]]],
            dtype=torch.float64,
        )
    )

    teacher = map_teacher_sampled_log_probs(
        raw_sampled_log_probs=raw,
        teacher_eos_log_probs=teacher_eos,
        sampled_ids=sampled_ids,
        mode="semantic_class",
        semantic_eos_token_ids=[1, 3, 4],
        epsilon=1e-12,
    )
    torch.testing.assert_close(teacher.exp(), torch.tensor([[0.60, 0.40, 0.40, 0.55]], dtype=torch.float64))

    logits = torch.log(
        torch.tensor([[[0.05, 0.10, 0.40, 0.20, 0.25]] * 4], dtype=torch.float64)
    )
    student = student_policy_log_probs(
        logits=logits,
        sampled_ids=sampled_ids,
        raw_sampled_log_probs=raw,
        mode="semantic_class",
        semantic_eos_token_ids=[1, 3, 4],
    )
    torch.testing.assert_close(student.exp(), torch.tensor([[0.55, 0.55, 0.55, 0.55]], dtype=torch.float64))


def test_semantic_class_accepts_a_single_terminal_token():
    sampled_ids = torch.tensor([[1, 2]])
    raw = torch.log(torch.tensor([[0.25, 0.50]], dtype=torch.float64))
    teacher_eos = torch.log(torch.tensor([[[0.25], [0.10]]], dtype=torch.float64))

    teacher = map_teacher_sampled_log_probs(
        raw_sampled_log_probs=raw,
        teacher_eos_log_probs=teacher_eos,
        sampled_ids=sampled_ids,
        mode="semantic_class",
        semantic_eos_token_ids=[1],
        epsilon=1e-12,
    )

    torch.testing.assert_close(teacher.exp(), torch.tensor([[0.25, 0.50]], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["teacher_map", "canonical"])
def test_pairwise_only_modes_reject_three_semantic_eos_ids(mode):
    with pytest.raises(ValueError, match="exactly two"):
        student_policy_log_probs(
            logits=torch.zeros(1, 1, 5),
            sampled_ids=torch.tensor([[1]]),
            raw_sampled_log_probs=torch.zeros(1, 1),
            mode=mode,
            semantic_eos_token_ids=[1, 3, 4],
        )


def test_fixed_token_log_probs_match_full_log_softmax_without_materializing_it():
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    selected = gather_fixed_token_log_probs(logits, [3, 1])

    torch.testing.assert_close(selected, torch.log_softmax(logits, dim=-1)[..., [3, 1]])
