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

import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.trainer.ppo import metric_utils
from verl.utils import torch_functional as verl_F


def test_eos_diagnostic_metadata_is_initialized_independently_of_reward_model_execution():
    batch = SimpleNamespace(meta_info={})
    rollout_config = {
        "eos_mode": "canonical",
        "semantic_eos_token_ids": [151643, 151645],
        "eos_mapping_epsilon": 1e-12,
    }

    eos_mode, token_ids, epsilon = metric_utils.attach_eos_semantics_metadata(batch, rollout_config)

    assert eos_mode == "canonical"
    assert token_ids == [151643, 151645]
    assert epsilon == 1e-12
    assert batch.meta_info == {
        "eos_mode": "canonical",
        "semantic_eos_token_ids": [151643, 151645],
        "eos_mapping_epsilon": 1e-12,
    }


def test_gather_log_probs_for_token_ids_preserves_requested_order():
    gather_fn = getattr(verl_F, "gather_log_probs_for_token_ids", None)
    assert gather_fn is not None, "tracked-token log-prob gathering is not implemented"

    log_probs = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)

    actual = gather_fn(log_probs, token_ids=[4, 1])

    expected = torch.stack((log_probs[..., 4], log_probs[..., 1]), dim=-1)
    torch.testing.assert_close(actual, expected)


def test_tracked_token_metrics_average_each_sequence_and_use_its_last_valid_position():
    compute_fn = getattr(metric_utils, "compute_tracked_token_probability_metrics", None)
    assert compute_fn is not None, "tracked-token probability metrics are not implemented"

    probabilities = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.9, 0.9], [0.9, 0.9]],
            [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [0.9, 0.9]],
            [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    metrics = compute_fn(
        tracked_token_log_probs=probabilities.log(),
        response_mask=response_mask,
        token_names=["im_end_151645", "endoftext_151643"],
    )

    assert metrics == pytest.approx(
        {
            "student_eos_prob/im_end_151645/sequence_mean": 0.45,
            "student_eos_prob/im_end_151645/last_token": 0.6,
            "student_eos_prob/endoftext_151643/sequence_mean": 0.55,
            "student_eos_prob/endoftext_151643/last_token": 0.7,
        }
    )


def test_eos_diagnostics_cover_lengths_finish_probs_sampling_and_post_e2_tokens():
    responses = torch.tensor(
        [
            [11, 0, 0, 0],
            [7, 13, 0, 0],
            [7, 13, 9, 0],
            [7, 8, 9, 10],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    advantages = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    student_probs = torch.tensor([0.1, 0.2]).expand(4, 4, 2)
    teacher_probs = torch.tensor([0.3, 0.4]).expand(4, 4, 2)

    metrics = metric_utils.compute_eos_diagnostic_metrics(
        responses=responses,
        response_mask=response_mask,
        advantages=advantages,
        student_eos_log_probs=student_probs.log(),
        teacher_eos_log_probs=teacher_probs.log(),
        eos_token_ids=[11, 13],
        token_names=["e1", "e2"],
    )

    expected_legacy_metrics = {
            "response_length/median": 2.5,
            "response_length/p95": 3.85,
            "finish_reason/e1_fraction": 0.25,
            "finish_reason/e2_fraction": 0.25,
            "finish_reason/max_tokens_fraction": 0.25,
            "finish_reason/other_fraction": 0.25,
            "sampled_eos/e1/count": 1.0,
            "sampled_eos/e1/mean_advantage": 1.0,
            "sampled_eos/e2/count": 2.0,
            "sampled_eos/e2/mean_advantage": 3.0,
            "eos/e2/tokens_after_mean": 0.5,
            "eos/e2/sequence_fraction": 0.5,
            "student_eos_prob/e1/sequence_mean": 0.1,
            "student_eos_prob/e1/last_token": 0.1,
            "student_eos_prob/e2/sequence_mean": 0.2,
            "student_eos_prob/e2/last_token": 0.2,
            "student_eos_prob/sum/sequence_mean": 0.3,
            "student_eos_prob/sum/last_token": 0.3,
            "teacher_eos_prob/e1/sequence_mean": 0.3,
            "teacher_eos_prob/e1/last_token": 0.3,
            "teacher_eos_prob/e2/sequence_mean": 0.4,
            "teacher_eos_prob/e2/last_token": 0.4,
            "teacher_eos_prob/sum/sequence_mean": 0.7,
            "teacher_eos_prob/sum/last_token": 0.7,
    }
    for name, expected in expected_legacy_metrics.items():
        assert metrics[name] == pytest.approx(expected)
    assert metrics["response_length/mean"] == pytest.approx(2.5)
    assert metrics["response_length/min"] == pytest.approx(1.0)
    assert metrics["response_length/max"] == pytest.approx(4.0)
    assert metrics["finish_reason/token_11_fraction"] == pytest.approx(0.25)
    assert metrics["finish_reason/token_13_fraction"] == pytest.approx(0.25)
    assert metrics["finish_reason/semantic_terminal_fraction"] == pytest.approx(0.5)


def test_eos_diagnostics_support_three_terminal_ids_with_dynamic_keys():
    responses = torch.tensor([[11, 0], [13, 0], [17, 0], [9, 8]])
    response_mask = torch.tensor([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=torch.bool)
    probabilities = torch.tensor([0.1, 0.2, 0.3]).expand(4, 2, 3)

    metrics = metric_utils.compute_eos_diagnostic_metrics(
        responses=responses,
        response_mask=response_mask,
        advantages=torch.zeros_like(responses, dtype=torch.float32),
        student_eos_log_probs=probabilities.log(),
        teacher_eos_log_probs=probabilities.log(),
        eos_token_ids=[11, 13, 17],
        token_names=["token_11", "token_13", "token_17"],
    )

    assert metrics["finish_reason/token_11_fraction"] == pytest.approx(0.25)
    assert metrics["finish_reason/token_13_fraction"] == pytest.approx(0.25)
    assert metrics["finish_reason/token_17_fraction"] == pytest.approx(0.25)
    assert metrics["finish_reason/semantic_terminal_fraction"] == pytest.approx(0.75)
    assert metrics["teacher_eos_prob/token_17/last_token"] == pytest.approx(0.3)
    assert metrics["teacher_eos_prob/sum/last_token"] == pytest.approx(0.6)


def test_teacher_eos_records_keep_all_terminal_columns_and_finished_token_ids():
    responses = torch.tensor([[5, 0], [7, 0], [9, 8]])
    response_mask = torch.tensor([[1, 0], [1, 0], [1, 1]], dtype=torch.bool)
    teacher_probs = torch.tensor([0.1, 0.2, 0.3]).expand(3, 2, 3).clone()

    records = metric_utils.extract_teacher_eos_at_student_eos(
        responses=responses,
        response_mask=response_mask,
        teacher_eos_log_probs=teacher_probs.log(),
        terminal_token_ids=[5, 7, 11],
    )

    assert records["teacher_eos_probs"].shape == (2, 3)
    assert records["teacher_eos_shares"].shape == (2, 3)
    np.testing.assert_array_equal(records["sampled_terminal_id"], [5, 7])


def test_teacher_eos_records_select_only_sequences_that_finish_with_student_eos():
    extract_fn = getattr(metric_utils, "extract_teacher_eos_at_student_eos", None)
    assert extract_fn is not None, "conditional teacher-EOS extraction is not implemented"

    responses = torch.tensor(
        [
            [5, 0, 0],
            [2, 5, 0],
            [5, 2, 0],
            [0, 0, 0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.bool,
    )
    teacher_probs = torch.full((4, 3, 2), 0.25)
    teacher_probs[0, 0] = torch.tensor([0.01, 0.09])
    teacher_probs[1, 1] = torch.tensor([0.20, 0.30])
    teacher_probs[2, 1] = torch.tensor([0.40, 0.10])

    records = extract_fn(
        responses=responses,
        response_mask=response_mask,
        teacher_eos_log_probs=teacher_probs.log(),
        student_eos_token_id=5,
    )

    np.testing.assert_array_equal(records["response_length"], np.array([1, 2]))
    np.testing.assert_allclose(records["teacher_prob_e1"], [0.01, 0.20], rtol=1e-6)
    np.testing.assert_allclose(records["teacher_prob_e2"], [0.09, 0.30], rtol=1e-6)
    np.testing.assert_allclose(records["teacher_eos_mass"], [0.10, 0.50], rtol=1e-6)
    np.testing.assert_allclose(records["teacher_e1_share"], [0.10, 0.40], rtol=1e-6)
    np.testing.assert_allclose(records["teacher_e2_share"], [0.90, 0.60], rtol=1e-6)
    np.testing.assert_allclose(records["logprob_e2_minus_e1"], np.log([9.0, 1.5]), rtol=1e-6)


def test_teacher_eos_records_are_saved_as_one_compressed_npz_per_step(tmp_path):
    save_fn = getattr(metric_utils, "save_teacher_eos_diagnostic_npz", None)
    assert save_fn is not None, "compressed teacher-EOS diagnostic output is not implemented"
    records = {
        "response_length": np.array([3], dtype=np.int64),
        "teacher_prob_e1": np.array([0.02], dtype=np.float32),
        "teacher_prob_e2": np.array([0.18], dtype=np.float32),
        "teacher_eos_mass": np.array([0.20], dtype=np.float32),
        "teacher_e1_share": np.array([0.10], dtype=np.float32),
        "teacher_e2_share": np.array([0.90], dtype=np.float32),
        "logprob_e2_minus_e1": np.array([np.log(9.0)], dtype=np.float32),
    }

    output_path = save_fn(
        output_dir=tmp_path,
        global_step=4,
        eos_token_ids=[151643, 151645],
        records=records,
    )

    assert output_path == tmp_path / "step_0004_teacher_eos_at_student_eos.npz"
    with np.load(output_path) as saved:
        assert saved["global_step"].item() == 4
        np.testing.assert_array_equal(saved["eos_token_ids"], [151643, 151645])
        np.testing.assert_array_equal(saved["response_length"], [3])
        np.testing.assert_allclose(saved["teacher_prob_e1"], [0.02])
        np.testing.assert_allclose(saved["teacher_prob_e2"], [0.18])

    with (tmp_path / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "global_step": "4",
            "count": "1",
            "teacher_prob_e1_mean": "0.019999999552965164",
            "teacher_prob_e2_mean": "0.18000000715255737",
            "teacher_eos_mass_mean": "0.20000000298023224",
            "teacher_e1_share_mean": "0.10000000149011612",
            "teacher_e2_share_mean": "0.8999999761581421",
            "logprob_e2_minus_e1_mean": str(float(np.float32(np.log(9.0)))),
            "response_length_mean": "3.0",
        }
    ]
