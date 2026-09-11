import pytest

from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    merge_eos_sampling_overrides,
    resolve_eos_sampling_overrides,
)


def test_rollout_config_defaults_to_the_existing_baseline_semantics():
    config = RolloutConfig(name="vllm", prompt_length=32, response_length=64)

    assert config.eos_mode == "baseline"
    assert config.semantic_eos_token_ids == []
    assert config.eos_mapping_epsilon == 1e-12


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("baseline", {}),
        ("two_stop", {"stop_token_ids": [151643, 151645]}),
        ("teacher_map", {"stop_token_ids": [151643]}),
        ("semantic_class", {"stop_token_ids": [151643, 151645]}),
        ("canonical", {"stop_token_ids": [151643], "bad_words": ["<|im_end|>"]}),
    ],
)
def test_eos_modes_resolve_exact_vllm_sampling_overrides(mode, expected):
    assert (
        resolve_eos_sampling_overrides(
            mode=mode,
            semantic_eos_token_ids=[151643, 151645],
            canonical_bad_word="<|im_end|>",
        )
        == expected
    )


def test_canonical_mode_requires_a_single_token_bad_word_for_e2():
    with pytest.raises(ValueError, match="canonical_bad_word"):
        resolve_eos_sampling_overrides(
            mode="canonical",
            semantic_eos_token_ids=[151643, 151645],
            canonical_bad_word=None,
        )


def test_pairwise_only_modes_require_exactly_two_distinct_semantic_eos_ids():
    with pytest.raises(ValueError, match="exactly two distinct"):
        resolve_eos_sampling_overrides(
            mode="teacher_map",
            semantic_eos_token_ids=[151643],
            canonical_bad_word=None,
        )


@pytest.mark.parametrize("mode", ["two_stop", "semantic_class"])
def test_stop_set_modes_accept_three_distinct_semantic_eos_ids(mode):
    assert resolve_eos_sampling_overrides(
        mode=mode,
        semantic_eos_token_ids=[1, 106, 128],
        canonical_bad_word=None,
    ) == {"stop_token_ids": [1, 106, 128]}


@pytest.mark.parametrize("mode", ["two_stop", "semantic_class"])
def test_stop_set_modes_accept_one_semantic_eos_id(mode):
    assert resolve_eos_sampling_overrides(
        mode=mode,
        semantic_eos_token_ids=[1],
        canonical_bad_word=None,
    ) == {"stop_token_ids": [1]}


def test_canonical_override_preserves_existing_bad_words():
    kwargs = {"bad_words": ["existing"], "stop_token_ids": [99]}

    merge_eos_sampling_overrides(
        kwargs,
        {"bad_words": ["<|im_end|>"], "stop_token_ids": [151643]},
    )

    assert kwargs["bad_words"] == ["existing", "<|im_end|>"]
    assert kwargs["stop_token_ids"] == [151643]
