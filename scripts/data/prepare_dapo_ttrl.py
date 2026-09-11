#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
TTRL_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def convert_prompt(messages: list[dict[str, str]], prompt_format: str = "ttrl") -> list[dict[str, str]]:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("expected exactly one user message")

    content = messages[0]["content"]
    if not content.startswith(DAPO_PREFIX) or not content.endswith(DAPO_SUFFIX):
        raise ValueError("prompt does not match the expected DAPO wrapper")

    problem = content[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)].strip()
    converted_message = dict(messages[0])
    if prompt_format == "ttrl":
        converted_message["content"] = problem + TTRL_SUFFIX
    elif prompt_format == "raw-question":
        converted_message["content"] = problem
    else:
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    return [converted_message]


def convert_dataset(input_path: Path, output_path: Path, prompt_format: str = "ttrl") -> int:
    table = pq.read_table(input_path)
    prompt_index = table.schema.get_field_index("prompt")
    if prompt_index < 0:
        raise ValueError("input parquet has no prompt column")

    prompt_field = table.schema.field(prompt_index)
    converted_prompts = [
        convert_prompt(messages, prompt_format=prompt_format) for messages in table.column(prompt_index).to_pylist()
    ]
    converted_table = table.set_column(
        prompt_index,
        prompt_field,
        pa.array(converted_prompts, type=prompt_field.type),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(converted_table, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return converted_table.num_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DAPO prompts to TTRL or raw-question format.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-format", choices=("ttrl", "raw-question"), default="ttrl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_count = convert_dataset(args.input, args.output, prompt_format=args.prompt_format)
    print(f"wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
