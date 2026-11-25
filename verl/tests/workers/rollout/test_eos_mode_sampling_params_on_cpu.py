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
        ("teacher_map", {"stop_token_ids": [151643], "ignore_eos": True}),
        ("semantic_class", {"stop_token_ids": [151643, 151645]}),
        ("canonical", {"stop_token_ids": [151643], "ignore_eos": True, "bad_words": ["<|im_end|>"]}),
    ],
)
def test_eos_modes_resolve_exact_vllm_sampling_overrides(mode, expected):
    assert (
        resolve_eos_sampling_overrides(
            mode=mode,
            semantic_eos_token_ids=[151643, 151645],
            canonical_bad_words=["<|im_end|>"],
        )
        == expected
    )


def test_canonical_mode_requires_bad_words_for_every_noncanonical_eos():
    with pytest.raises(ValueError, match="one bad word for each noncanonical EOS"):
        resolve_eos_sampling_overrides(
            mode="canonical",
            semantic_eos_token_ids=[151643, 151645],
        )


def test_mapping_modes_require_at_least_two_distinct_semantic_eos_ids():
    with pytest.raises(ValueError, match="at least two distinct"):
        resolve_eos_sampling_overrides(
            mode="teacher_map",
            semantic_eos_token_ids=[151643],
        )


@pytest.mark.parametrize("mode", ["two_stop", "semantic_class"])
def test_stop_set_modes_accept_three_distinct_semantic_eos_ids(mode):
    assert resolve_eos_sampling_overrides(
        mode=mode,
        semantic_eos_token_ids=[1, 106, 128],
    ) == {"stop_token_ids": [1, 106, 128]}


@pytest.mark.parametrize("mode", ["two_stop", "semantic_class"])
def test_stop_set_modes_accept_one_semantic_eos_id(mode):
    assert resolve_eos_sampling_overrides(
        mode=mode,
        semantic_eos_token_ids=[1],
    ) == {"stop_token_ids": [1]}


def test_llama_canonical_disables_default_eos_and_blocks_both_noncanonical_tokens():
    bad_words = ["<|eom_id|>", "<|eot_id|>"]
    assert resolve_eos_sampling_overrides(
        mode="canonical",
        semantic_eos_token_ids=[128001, 128008, 128009],
        canonical_bad_words=bad_words,
    ) == {"stop_token_ids": [128001], "ignore_eos": True, "bad_words": bad_words}


def test_canonical_override_preserves_existing_bad_words():
    kwargs = {"bad_words": ["existing"], "stop_token_ids": [99]}

    merge_eos_sampling_overrides(
        kwargs,
        {"bad_words": ["<|im_end|>"], "stop_token_ids": [151643]},
    )

    assert kwargs["bad_words"] == ["existing", "<|im_end|>"]
    assert kwargs["stop_token_ids"] == [151643]
