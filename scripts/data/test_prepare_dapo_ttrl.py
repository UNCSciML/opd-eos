import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).with_name("prepare_dapo_ttrl.py")


def test_cli_converts_dapo_prompt_to_single_brace_ttrl_and_preserves_fields(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    row = {
        "data_source": "unit-test",
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. The last line of your response should be of the "
                    "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
                    "What is 2+2?\n\n"
                    'Remember to put your answer on its own line after "Answer:".'
                ),
            }
        ],
        "ability": "math",
        "reward_model": {"ground_truth": "4", "style": "rule"},
        "extra_info": {"index": "example-0"},
    }
    pq.write_table(pa.Table.from_pylist([row]), source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    converted = pq.read_table(output).to_pylist()[0]
    assert converted == {
        **row,
        "prompt": [
            {
                "role": "user",
                "content": (
                    "What is 2+2? Please reason step by step, and put your final answer within \\boxed{}."
                ),
            }
        ],
    }


def test_cli_converts_dapo_prompt_to_raw_question_and_preserves_fields(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    row = {
        "data_source": "unit-test",
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. The last line of your response should be of the "
                    "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
                    "What is 2+2?\n\n"
                    'Remember to put your answer on its own line after "Answer:".'
                ),
            }
        ],
        "ability": "math",
        "reward_model": {"ground_truth": "4", "style": "rule"},
        "extra_info": {"index": "example-0"},
    }
    pq.write_table(pa.Table.from_pylist([row]), source)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--prompt-format",
            "raw-question",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    converted = pq.read_table(output).to_pylist()[0]
    assert converted == {
        **row,
        "prompt": [{"role": "user", "content": "What is 2+2?"}],
    }
