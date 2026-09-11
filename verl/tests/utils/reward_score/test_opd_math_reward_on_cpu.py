from verl.utils.reward_score.opd_math import compute_score


def test_opd_math_reward_semantically_grades_boxed_and_answer_line_outputs():
    assert compute_score("work\n\\boxed{4}", "4")["score"] == 1.0
    assert compute_score("work\nAnswer: 4", "4")["score"] == 1.0


def test_opd_math_reward_rejects_outputs_without_an_extractable_answer():
    result = compute_score("work\n4", "4")

    assert result["score"] == 0.0
    assert result["format_score"] == 0.0
