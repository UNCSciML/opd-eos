#!/usr/bin/env python3
"""Persist and read generation semantics shared by OPD train and eval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EOS_MODES = {"baseline", "two_stop", "teacher_map", "semantic_class", "canonical"}
PROMPT_TEMPLATES = {"dapo", "ttrl", "eopd"}


def normalize_token_ids(value: int | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result: list[int] = []
    for token_id in values:
        token_id = int(token_id)
        if token_id not in result:
            result.append(token_id)
    return result


def ordered_union(*token_id_groups: list[int]) -> list[int]:
    return normalize_token_ids([token_id for group in token_id_groups for token_id in group])


def parse_token_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return normalize_token_ids([int(part.strip()) for part in value.split(",") if part.strip()])


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def build_run_semantics(
    *,
    model_path: str,
    model_eos_token_ids: int | list[int] | tuple[int, ...] | None,
    additional_eos_token_ids: list[int],
    enable_thinking: bool,
    eos_mode: str,
    semantic_eos_token_ids: list[int],
    eos_mapping_epsilon: float,
    prompt_template: str,
    train_dataset: str,
    teacher_model_path: str | None = None,
    teacher_eos_token_ids: int | list[int] | tuple[int, ...] | None = None,
    chat_template_model: str | None = None,
    chat_template_sha256: str | None = None,
    thinking_supported: bool | None = None,
) -> dict[str, Any]:
    student_ids = normalize_token_ids(model_eos_token_ids)
    teacher_ids = normalize_token_ids(teacher_eos_token_ids)
    if not student_ids:
        raise ValueError("The student model has no EOS token ids")
    eos_mode = eos_mode.strip().lower()
    if eos_mode not in EOS_MODES:
        raise ValueError(f"Unsupported EOS mode: {eos_mode}")
    semantic_ids = normalize_token_ids(semantic_eos_token_ids) or ordered_union(student_ids, teacher_ids)
    if not semantic_ids:
        raise ValueError("At least one semantic terminal token id is required")
    if eos_mode in {"teacher_map", "canonical"} and len(student_ids) != 1:
        raise ValueError(f"{eos_mode} requires exactly one student-native EOS token id")
    if eos_mode in {"teacher_map", "canonical"} and len(semantic_ids) != 2:
        raise ValueError(f"{eos_mode} requires exactly two distinct semantic terminal token ids")
    if student_ids[0] != semantic_ids[0]:
        raise ValueError("The first semantic EOS id must be the model's primary EOS id")
    prompt_template = prompt_template.strip().lower()
    if prompt_template not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt template: {prompt_template}")
    if eos_mapping_epsilon <= 0:
        raise ValueError("EOS mapping epsilon must be positive")

    if eos_mode in {"two_stop", "semantic_class"}:
        additional_ids = ordered_union(
            normalize_token_ids(additional_eos_token_ids),
            [token_id for token_id in semantic_ids if token_id not in student_ids],
        )
        rollout_stop_ids = semantic_ids
    elif eos_mode == "baseline":
        additional_ids = []
        rollout_stop_ids = student_ids
    else:
        additional_ids = []
        rollout_stop_ids = [semantic_ids[0]]
    effective_ids = ordered_union(student_ids, additional_ids)
    blocked_ids = [semantic_ids[1]] if eos_mode == "canonical" else []
    return {
        "schema_version": 3,
        "model_path": model_path,
        "student_model_path": model_path,
        "teacher_model_path": teacher_model_path,
        "chat_template_source": chat_template_model or model_path,
        "chat_template_sha256": chat_template_sha256,
        "thinking_supported": thinking_supported,
        "enable_thinking": enable_thinking,
        "prompt_template": prompt_template,
        "train_dataset": train_dataset,
        "eos_mode": eos_mode,
        "semantic_terminal_token_ids": semantic_ids,
        "semantic_eos_token_ids": semantic_ids,
        "eos_mapping_epsilon": eos_mapping_epsilon,
        "student_native_eos_token_ids": student_ids,
        "teacher_native_eos_token_ids": teacher_ids,
        "model_eos_token_ids": student_ids,
        "additional_eos_token_ids": additional_ids,
        "effective_eos_token_ids": effective_ids,
        "rollout_stop_token_ids": rollout_stop_ids,
        "blocked_token_ids": blocked_ids,
        "tracked_token_ids": semantic_ids,
        "tracked_token_names": [f"token_{token_id}" for token_id in semantic_ids],
    }


def discover_model_eos_token_ids(model_path: str) -> list[int]:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    try:
        generation_ids = normalize_token_ids(
            GenerationConfig.from_pretrained(model_path, trust_remote_code=True).eos_token_id
        )
    except (OSError, ValueError):
        generation_ids = []
    if generation_ids:
        return generation_ids

    config_ids = normalize_token_ids(AutoConfig.from_pretrained(model_path, trust_remote_code=True).eos_token_id)
    if config_ids:
        return config_ids

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return normalize_token_ids(tokenizer.eos_token_id)


def discover_chat_template(model_path: str) -> tuple[str, bool]:
    from transformers import AutoTokenizer

    template = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True).chat_template
    if not isinstance(template, str) or not template:
        raise ValueError(f"Tokenizer {model_path!r} does not define a chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest(), "enable_thinking" in template


def write_semantics(args: argparse.Namespace) -> None:
    template_source = args.chat_template_model or args.model
    template_sha256, thinking_supported = discover_chat_template(template_source)
    semantics = build_run_semantics(
        model_path=args.model,
        model_eos_token_ids=discover_model_eos_token_ids(args.model),
        teacher_model_path=args.teacher_model,
        teacher_eos_token_ids=(
            discover_model_eos_token_ids(args.teacher_model) if args.teacher_model else []
        ),
        chat_template_model=template_source,
        chat_template_sha256=template_sha256,
        thinking_supported=thinking_supported,
        additional_eos_token_ids=parse_token_ids(args.additional_eos_token_ids),
        enable_thinking=parse_bool(args.enable_thinking),
        eos_mode=args.eos_mode,
        semantic_eos_token_ids=parse_token_ids(args.semantic_eos_token_ids),
        eos_mapping_epsilon=args.eos_mapping_epsilon,
        prompt_template=args.prompt_template,
        train_dataset=args.train_dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(semantics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(semantics, sort_keys=True))


def get_semantics_field(args: argparse.Namespace) -> None:
    semantics = json.loads(args.path.read_text(encoding="utf-8"))
    if args.field not in semantics:
        raise KeyError(f"Missing field {args.field!r} in {args.path}")
    value = semantics[args.field]
    if isinstance(value, list):
        print(",".join(str(item) for item in value))
    elif isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write a run semantics manifest")
    write_parser.add_argument("--model", required=True)
    write_parser.add_argument("--teacher-model")
    write_parser.add_argument("--chat-template-model")
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--enable-thinking", default="false")
    write_parser.add_argument("--additional-eos-token-ids", default="")
    write_parser.add_argument("--eos-mode", choices=sorted(EOS_MODES), default="baseline")
    write_parser.add_argument("--semantic-eos-token-ids", default="")
    write_parser.add_argument("--eos-mapping-epsilon", type=float, default=1e-12)
    write_parser.add_argument("--prompt-template", choices=sorted(PROMPT_TEMPLATES), default="ttrl")
    write_parser.add_argument("--train-dataset", required=True)
    write_parser.set_defaults(func=write_semantics)

    get_parser = subparsers.add_parser("get", help="Print one field from a run semantics manifest")
    get_parser.add_argument("--path", type=Path, required=True)
    get_parser.add_argument("--field", required=True)
    get_parser.set_defaults(func=get_semantics_field)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
