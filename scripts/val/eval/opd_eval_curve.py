#!/usr/bin/env python3
"""Evaluate OPD checkpoints every N steps with the JustRL-style pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import extract_answer, grade_answer_verl
from verl.utils.tokenizer import load_chat_template_text, resolve_chat_template_kwargs


PROMPT_TEMPLATES = {
    "dapo": (
        "Solve the following math problem step by step. The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        "{problem}\n\nRemember to put your answer on its own line after \"Answer:\"."
    ),
    "ttrl": "{problem} Please reason step by step, and put your final answer within \\boxed{{}}.",
    "eopd": "{problem}",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ModelSpec:
    step: int
    label: str
    path: Path
    needs_merge: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval curve for OPD checkpoints.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--tasks", default="AIME24,AIME25,AMC23")
    parser.add_argument("--prompt-template", choices=sorted(PROMPT_TEMPLATES), default="ttrl")
    parser.add_argument(
        "--chat-template-model",
        default=None,
        help="Optional compatible model used only as the source of chat-template text.",
    )
    parser.add_argument("--steps", default="")
    parser.add_argument("--step-stride", type=int, default=20)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--length-tokenizer", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--batched-n-sampling",
        action="store_true",
        help="Use one vLLM request per prompt with SamplingParams.n instead of looping over rollout ids.",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help="Comma-separated EOS/stop token ids. Defaults to the evaluated model tokenizer EOS.",
    )
    parser.add_argument(
        "--blocked-token-ids",
        default="",
        help="Comma-separated token ids excluded from sampling; empty keeps the model distribution unchanged.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="OPD Length Inflation")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="opd_eval_curve")
    return parser.parse_args()


def task_specs(data_dir: Path, task_names: str) -> list[TaskSpec]:
    specs = []
    for name in [item.strip() for item in task_names.split(",") if item.strip()]:
        specs.append(TaskSpec(name=name, path=data_dir / name / "test.parquet"))
    return specs


def load_samples(filepath: Path, prompt_template: str = "ttrl") -> list[dict[str, Any]]:
    df = pd.read_parquet(filepath)
    if any(name in str(filepath) for name in ("BRUMO25", "CMIMC25", "HMMT25")):
        samples = [
            {
                "example_id": i,
                "prompt": str(df.at[i, "problem"]).strip(),
                "answer": str(df.at[i, "answer"]).strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": str(df.at[i, "prompt"][0]["content"]).strip(),
                "answer": str(df.at[i, "reward_model"]["ground_truth"]).strip(),
            }
            for i in range(len(df))
        ]

    for sample in samples:
        sample["prompt"] = PROMPT_TEMPLATES[prompt_template].format(problem=sample["prompt"])
    return samples


def parse_step_list(steps: str) -> list[int] | None:
    if not steps:
        return None
    parsed = []
    for part in steps.split(","):
        part = part.strip()
        if part:
            parsed.append(int(part))
    return sorted(set(parsed))


def discover_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    requested = parse_step_list(args.steps)
    if requested is None:
        requested = [0]
        for child in args.checkpoint_root.glob("global_step_*"):
            match = re.fullmatch(r"global_step_(\d+)", child.name)
            if not match:
                continue
            step = int(match.group(1))
            if step % args.step_stride != 0:
                continue
            if args.max_step is not None and step > args.max_step:
                continue
            requested.append(step)

    specs: list[ModelSpec] = []
    for step in sorted(set(requested)):
        if args.max_step is not None and step > args.max_step:
            continue
        label = f"step_{step:04d}"
        if step == 0:
            specs.append(ModelSpec(step=0, label=label, path=args.base_model, needs_merge=False))
            continue

        actor_dir = args.checkpoint_root / f"global_step_{step}" / "actor"
        if not actor_dir.exists():
            print(f"[warn] missing checkpoint for step {step}: {actor_dir}", flush=True)
            continue
        specs.append(ModelSpec(step=step, label=label, path=actor_dir, needs_merge=True))

    return specs


def is_hf_model_dir(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def ensure_merged_model(model: ModelSpec, merged_root: Path) -> Path:
    if not model.needs_merge:
        return model.path

    target_dir = merged_root / model.label
    if is_hf_model_dir(target_dir):
        print(f"[merge] reuse {model.label}: {target_dir}", flush=True)
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(model.path),
        "--target_dir",
        str(target_dir),
    ]
    print(f"[merge] {model.label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return target_dir


def parse_gpu_ids(gpu_ids: str) -> list[str]:
    if gpu_ids != "auto":
        return [item.strip() for item in gpu_ids.split(",") if item.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    count = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or "0")
    if count > 0:
        return [str(i) for i in range(count)]
    return ["0"]


def split_rollout_ids(n: int, num_workers: int) -> list[list[int]]:
    chunks = [[] for _ in range(num_workers)]
    for idx, rollout_id in enumerate(range(n)):
        chunks[idx % num_workers].append(rollout_id)
    return chunks


def split_task_payloads_by_prompt(task_payloads: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    chunks: list[dict[str, list[dict[str, Any]]]] = [
        {task["name"]: [] for task in task_payloads} for _ in range(num_workers)
    ]
    cursor = 0
    for task in task_payloads:
        for sample in task["samples"]:
            chunks[cursor % num_workers][task["name"]].append(sample)
            cursor += 1

    return [
        [
            {"name": task["name"], "samples": chunk[task["name"]]}
            for task in task_payloads
            if chunk[task["name"]]
        ]
        for chunk in chunks
    ]


def parse_stop_token_ids(value: str) -> list[int] | None:
    if not value.strip():
        return None
    result: list[int] = []
    for part in value.split(","):
        if not part.strip():
            continue
        token_id = int(part.strip())
        if token_id not in result:
            result.append(token_id)
    if not result:
        raise ValueError("--stop-token-ids did not contain any token ids")
    return result


def resolve_stop_token_ids(tokenizer: Any, explicit_ids: list[int] | None) -> list[int]:
    value = explicit_ids if explicit_ids is not None else tokenizer.eos_token_id
    values = value if isinstance(value, (list, tuple)) else [value]
    result: list[int] = []
    for token_id in values:
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id not in result:
            result.append(token_id)
    if not result:
        raise ValueError("No EOS/stop token ids were configured for evaluation")
    return result


def resolve_blocked_token_bad_words(tokenizer: Any, blocked_token_ids: list[int] | None) -> list[str]:
    bad_words: list[str] = []
    for token_id in blocked_token_ids or []:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if not isinstance(token, str) or tokenizer.encode(token, add_special_tokens=False) != [int(token_id)]:
            raise ValueError(f"Blocked token {token_id} must round-trip as one tokenizer token")
        if token not in bad_words:
            bad_words.append(token)
    return bad_words


def build_generation_row(sample: dict[str, Any], rollout_id: int, completion: Any) -> dict[str, Any]:
    return {
        "example_id": sample["example_id"],
        "prompt": sample["prompt"],
        "answer": sample["answer"],
        "seed": rollout_id,
        "response": completion.text,
        "_response_token_ids": [int(token_id) for token_id in completion.token_ids],
    }


def generate_batched_requests(
    *,
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    task_payloads: list[dict[str, Any]],
    enable_thinking: bool,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    stop_token_ids: list[int] | None,
    blocked_token_ids: list[int] | None,
    chat_template: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Submit one prompt batch to a vLLM engine and expand its n completions."""
    results: dict[str, list[dict[str, Any]]] = {task["name"]: [] for task in task_payloads}
    flat_requests: list[tuple[str, dict[str, Any], str]] = []
    for task in task_payloads:
        for sample in task["samples"]:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}],
                **resolve_chat_template_kwargs(
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    enable_thinking=enable_thinking,
                    base_kwargs={"tokenize": False, "add_generation_prompt": True},
                ),
            )
            flat_requests.append((task["name"], sample, prompt))

    if not flat_requests:
        return results

    sampling_kwargs = dict(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        stop_token_ids=resolve_stop_token_ids(tokenizer, stop_token_ids),
    )
    bad_words = resolve_blocked_token_bad_words(tokenizer, blocked_token_ids)
    if bad_words:
        sampling_kwargs["bad_words"] = bad_words
    sampling = sampling_params_cls(**sampling_kwargs)
    outputs = llm.generate([item[2] for item in flat_requests], sampling, use_tqdm=False)
    for (task_name, sample, _), out in zip(flat_requests, outputs):
        for rollout_id, completion in enumerate(out.outputs):
            results[task_name].append(build_generation_row(sample, rollout_id, completion))
    return results


def worker_generate(args_tuple: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    (
        model_path,
        task_payloads,
        rollout_ids,
        gpu_id,
        enable_thinking,
        max_tokens,
        temperature,
        top_p,
        max_model_len,
        gpu_memory_utilization,
        trust_remote_code,
        seed_base,
        configured_stop_token_ids,
        configured_blocked_token_ids,
        chat_template,
    ) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    except ImportError:
        destroy_distributed_environment = None
        destroy_model_parallel = None

    results: dict[str, list[dict[str, Any]]] = {task["name"]: [] for task in task_payloads}
    llm = None
    try:
        print(f"[gpu {gpu_id}] loading {model_path}", flush=True)
        llm = LLM(
            model=str(model_path),
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        )
        tokenizer = llm.get_tokenizer()
        stop_token_ids = resolve_stop_token_ids(tokenizer, configured_stop_token_ids)
        bad_words = resolve_blocked_token_bad_words(tokenizer, configured_blocked_token_ids)

        for task in task_payloads:
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["prompt"]}],
                    **resolve_chat_template_kwargs(
                        tokenizer=tokenizer,
                        chat_template=chat_template,
                        enable_thinking=enable_thinking,
                        base_kwargs={"tokenize": False, "add_generation_prompt": True},
                    ),
                )
                for sample in task["samples"]
            ]

            for rollout_id in rollout_ids:
                sampling_kwargs = dict(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed_base + int(rollout_id),
                    stop_token_ids=stop_token_ids or None,
                )
                if bad_words:
                    sampling_kwargs["bad_words"] = bad_words
                sampling = SamplingParams(**sampling_kwargs)
                outputs = llm.generate(formatted_prompts, sampling, use_tqdm=False)
                for sample, out in zip(task["samples"], outputs):
                    results[task["name"]].append(build_generation_row(sample, rollout_id, out.outputs[0]))

    finally:
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[gpu {gpu_id}] cleanup done", flush=True)

    return results


def worker_generate_batched_n(args_tuple: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    (
        model_path,
        task_payloads,
        gpu_id,
        enable_thinking,
        n,
        max_tokens,
        temperature,
        top_p,
        max_model_len,
        gpu_memory_utilization,
        trust_remote_code,
        seed_base,
        configured_stop_token_ids,
        configured_blocked_token_ids,
        chat_template,
    ) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from vllm import LLM, SamplingParams

    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    except ImportError:
        destroy_distributed_environment = None
        destroy_model_parallel = None

    results: dict[str, list[dict[str, Any]]] = {task["name"]: [] for task in task_payloads}
    llm = None
    try:
        print(f"[gpu {gpu_id}] loading {model_path}", flush=True)
        llm = LLM(
            model=str(model_path),
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        )
        tokenizer = llm.get_tokenizer()
        prompt_count = sum(len(task["samples"]) for task in task_payloads)
        print(f"[gpu {gpu_id}] batched vLLM generate prompts={prompt_count} n={n}", flush=True)
        results = generate_batched_requests(
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=SamplingParams,
            task_payloads=task_payloads,
            enable_thinking=enable_thinking,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed_base,
            stop_token_ids=configured_stop_token_ids,
            blocked_token_ids=configured_blocked_token_ids,
            chat_template=chat_template,
        )

    finally:
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[gpu {gpu_id}] cleanup done", flush=True)

    return results


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def has_complete_generation_output(path: Path, expected_rows: int) -> bool:
    if count_lines(path) != expected_rows:
        return False
    token_path = path.with_suffix(".tokens.npz")
    if not token_path.is_file():
        return False
    try:
        with np.load(token_path, allow_pickle=False) as token_data:
            token_ids = token_data["token_ids"]
            offsets = token_data["offsets"]
            recorded_jsonl_sha256 = str(token_data["jsonl_sha256"].item())
            if offsets.shape != (expected_rows + 1,):
                return False
            if offsets[0] != 0 or offsets[-1] != len(token_ids):
                return False
            if np.any(offsets[1:] < offsets[:-1]):
                return False
            actual_jsonl_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if recorded_jsonl_sha256 != actual_jsonl_sha256:
                return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def prepare_eval_output_root(output_root: Path, config: dict[str, Any], replace: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "eval_run_config.json"
    existing_outputs = any(output_root.glob("step_*/*.jsonl"))
    existing_config: dict[str, Any] | None = None
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not replace:
                raise RuntimeError(f"Invalid evaluation config at {config_path}; use --replace to regenerate") from exc

    if existing_config is not None and existing_config != config and not replace:
        raise RuntimeError(
            f"Output root {output_root} contains different evaluation settings; "
            "choose a different --output-root or use --replace"
        )
    if existing_config is None and existing_outputs and not replace:
        raise RuntimeError(
            f"Output root {output_root} has generations but no trusted eval_run_config.json; "
            "choose a different --output-root or use --replace"
        )
    if existing_config == config:
        return

    temporary_path = config_path.with_name(f".{config_path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_path, config_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_eval_run_config(
    args: argparse.Namespace,
    task_payloads: list[dict[str, Any]],
    gpu_ids: list[str],
) -> dict[str, Any]:
    task_payload_json = json.dumps(task_payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "checkpoint_root": str(args.checkpoint_root),
        "base_model": str(args.base_model),
        "data_dir": str(args.data_dir),
        "tasks": args.tasks,
        "task_payload_sha256": hashlib.sha256(task_payload_json.encode("utf-8")).hexdigest(),
        "prompt_template": args.prompt_template,
        "chat_template_model": args.chat_template_model,
        "chat_template_sha256": args.chat_template_sha256,
        "enable_thinking": args.enable_thinking,
        "stop_token_ids": args.stop_token_ids or [],
        "blocked_token_ids": args.blocked_token_ids or [],
        "n": args.n,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "gpu_ids": gpu_ids,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batched_n_sampling": args.batched_n_sampling,
        "trust_remote_code": args.trust_remote_code,
    }


def write_generation_results(output_path: Path, rows: list[dict[str, Any]]) -> None:
    lengths = np.fromiter((len(row["_response_token_ids"]) for row in rows), dtype=np.int64, count=len(rows))
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    token_ids = np.empty(int(offsets[-1]), dtype=np.int32)

    with output_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            start, end = int(offsets[index]), int(offsets[index + 1])
            token_ids[start:end] = row["_response_token_ids"]
            json_row = {key: value for key, value in row.items() if key != "_response_token_ids"}
            f.write(json.dumps(json_row, ensure_ascii=False) + "\n")

    jsonl_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    np.savez(
        output_path.with_suffix(".tokens.npz"),
        token_ids=token_ids,
        offsets=offsets,
        jsonl_sha256=np.asarray(jsonl_sha256),
    )


def generate_for_model(
    model_path: Path,
    model_label: str,
    task_payloads: list[dict[str, Any]],
    gpu_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Path]:
    out_dir = args.output_root / model_label
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        task["name"]: out_dir / f"{task['name'].lower()}_t{args.temperature}_p{args.top_p}_n{args.n}-MNT{args.max_tokens}.jsonl"
        for task in task_payloads
    }

    expected = {task["name"]: len(task["samples"]) * args.n for task in task_payloads}
    if not args.replace and all(
        has_complete_generation_output(output_paths[name], expected[name]) for name in output_paths
    ):
        print(f"[generate] reuse complete outputs for {model_label}", flush=True)
        return output_paths

    if args.batched_n_sampling:
        task_chunks = split_task_payloads_by_prompt(task_payloads, len(gpu_ids))
        worker_args = [
            (
                str(model_path),
                task_chunks[idx],
                gpu_id,
                args.enable_thinking,
                args.n,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.max_model_len,
                args.gpu_memory_utilization,
                args.trust_remote_code,
                args.seed,
                args.stop_token_ids,
                args.blocked_token_ids,
                args.chat_template,
            )
            for idx, gpu_id in enumerate(gpu_ids)
            if task_chunks[idx]
        ]
        worker_fn = worker_generate_batched_n
    else:
        rollout_chunks = split_rollout_ids(args.n, len(gpu_ids))
        worker_args = [
            (
                str(model_path),
                task_payloads,
                rollout_chunks[idx],
                gpu_id,
                args.enable_thinking,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.max_model_len,
                args.gpu_memory_utilization,
                args.trust_remote_code,
                args.seed,
                args.stop_token_ids,
                args.blocked_token_ids,
                args.chat_template,
            )
            for idx, gpu_id in enumerate(gpu_ids)
            if rollout_chunks[idx]
        ]
        worker_fn = worker_generate

    all_results: dict[str, list[dict[str, Any]]] = {task["name"]: [] for task in task_payloads}
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
        futures = [executor.submit(worker_fn, item) for item in worker_args]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"generate {model_label}"):
            worker_results = fut.result()
            for task_name, rows in worker_results.items():
                all_results[task_name].extend(rows)

    for task_name, rows in all_results.items():
        rows.sort(key=lambda item: (int(item["example_id"]), int(item["seed"])))
        write_generation_results(output_paths[task_name], rows)
        print(f"[generate] {model_label}/{task_name}: {len(rows)} rows -> {output_paths[task_name]}", flush=True)

    return output_paths


def get_diverse_score(sequences: list[str], n: int = 4) -> float:
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            distinct_ngrams.add(tuple(tokens[i : i + n]))
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams else 0.0


def load_length_tokenizer(path: Path | None):
    if path is None:
        return None
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:
        print(f"[warn] failed to load length tokenizer {path}: {exc}", flush=True)
        return None


def grade_file(
    path: Path,
    length_tokenizer,
    prompt_template: str = "ttrl",
    stop_token_ids: list[int] | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    grouped: dict[int, dict[str, Any]] = {}
    token_sequences: list[list[int]] = []
    token_path = path.with_suffix(".tokens.npz")
    if token_path.exists():
        with np.load(token_path, allow_pickle=False) as token_data:
            flat_ids = token_data["token_ids"]
            offsets = token_data["offsets"]
            token_sequences = [
                flat_ids[int(offsets[index]) : int(offsets[index + 1])].astype(np.int64).tolist()
                for index in range(len(offsets) - 1)
            ]

    with path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            row = json.loads(line)
            example_id = int(row["example_id"])
            grouped.setdefault(example_id, {"answer": row["answer"], "responses": []})
            ids = token_sequences[row_index] if row_index < len(token_sequences) else None
            grouped[example_id]["responses"].append((str(row["response"]), ids))

    if not grouped:
        return {
            "mean_score": 0.0,
            "best_score": 0.0,
            "distinct_4gram": 0.0,
            "solve_none": 0,
            "solve_all": 0,
            "avg_output_length": 0.0,
            "semantic_accuracy": 0.0,
            "extraction_success": 0.0,
            "format_compliance": 0.0,
            "format_error_rollouts": 0,
            "num_examples": 0,
            "num_rollouts": 0,
        }

    avg_scores = []
    best_scores = []
    diverse_scores = []
    response_lengths = []
    extraction_successes = []
    format_compliances = []
    finish_tokens = []
    truncations = []
    tokens_after_e2 = []
    ordered_stop_ids = [int(token_id) for token_id in (stop_token_ids or [])]
    legacy_e2 = ordered_stop_ids[1] if len(ordered_stop_ids) == 2 else None
    num_rollouts = 0

    for example_id in sorted(grouped):
        item = grouped[example_id]
        response_entries = item["responses"]
        responses = [entry[0] for entry in response_entries]
        answer = item["answer"]
        scores = [bool(grade_answer_verl(response, answer, answer_format="auto")) for response in responses]
        avg_scores.append(sum(scores) / len(scores))
        best_scores.append(max(scores))
        diverse_scores.append(get_diverse_score(responses))
        num_rollouts += len(responses)
        for response, response_ids in response_entries:
            extraction_successes.append(extract_answer(response, answer_format="auto") is not None)
            if prompt_template == "ttrl":
                format_compliances.append("\\boxed" in response)
            elif prompt_template == "dapo":
                format_compliances.append(re.search(r"(?im)^\s*Answer:\s*.+$", response) is not None)
            else:
                format_compliances.append(extraction_successes[-1])

            if response_ids is not None:
                response_lengths.append(len(response_ids))
                final_token = response_ids[-1] if response_ids else None
                finish_tokens.append(final_token)
                configured_stops = set(stop_token_ids or [])
                truncations.append(
                    bool(max_tokens is not None and len(response_ids) >= max_tokens and final_token not in configured_stops)
                )
                if legacy_e2 is not None and legacy_e2 in response_ids:
                    tokens_after_e2.append(len(response_ids) - response_ids.index(legacy_e2) - 1)
            elif length_tokenizer is not None:
                response_lengths.append(len(length_tokenizer.encode(response)))
            else:
                response_lengths.append(len(response))

    mean_score = sum(avg_scores) / len(avg_scores)
    format_error_rollouts = num_rollouts - sum(format_compliances)
    response_lengths_array = np.asarray(response_lengths, dtype=np.float64)
    finish_denominator = max(len(finish_tokens), 1)
    metrics = {
        "mean_score": mean_score,
        "semantic_accuracy": mean_score,
        "best_score": sum(best_scores) / len(best_scores),
        "distinct_4gram": sum(diverse_scores) / len(diverse_scores),
        "solve_none": sum(1 for score in avg_scores if score == 0),
        "solve_all": sum(1 for score in avg_scores if score == 1),
        "avg_output_length": float(response_lengths_array.mean()) if response_lengths else 0.0,
        "output_length_mean": float(response_lengths_array.mean()) if response_lengths else 0.0,
        "output_length_median": float(np.median(response_lengths_array)) if response_lengths else 0.0,
        "output_length_p95": float(np.quantile(response_lengths_array, 0.95)) if response_lengths else 0.0,
        "extraction_success": sum(extraction_successes) / num_rollouts,
        "format_compliance": sum(format_compliances) / num_rollouts,
        "format_error_rollouts": format_error_rollouts,
        "truncation_rate": sum(truncations) / max(len(truncations), 1),
        "num_examples": len(grouped),
        "num_rollouts": num_rollouts,
    }
    for token_id in ordered_stop_ids:
        metrics[f"finish_token_{token_id}_fraction"] = (
            sum(finish_token == token_id for finish_token in finish_tokens) / finish_denominator
        )
    if len(ordered_stop_ids) == 2:
        metrics["finish_e1_fraction"] = metrics[f"finish_token_{ordered_stop_ids[0]}_fraction"]
        metrics["finish_e2_fraction"] = metrics[f"finish_token_{ordered_stop_ids[1]}_fraction"]
        metrics["tokens_after_e2_mean"] = (
            sum(tokens_after_e2) / len(tokens_after_e2) if tokens_after_e2 else 0.0
        )
    return metrics


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "grading_summary.json"
    csv_path = output_root / "grading_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[summary] wrote {json_path}", flush=True)
    print(f"[summary] wrote {csv_path}", flush=True)


def init_wandb(args: argparse.Namespace, specs: list[ModelSpec]):
    if not args.log_wandb:
        return None
    try:
        import wandb

        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config={
                "steps": [spec.step for spec in specs],
                "tasks": args.tasks,
                "n": args.n,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "checkpoint_root": str(args.checkpoint_root),
                "base_model": str(args.base_model),
                "enable_thinking": args.enable_thinking,
                "stop_token_ids": args.stop_token_ids,
                "blocked_token_ids": args.blocked_token_ids,
                "prompt_template": args.prompt_template,
            },
        )
    except Exception as exc:
        print(f"[warn] wandb init failed: {exc}", flush=True)
        return None


def main() -> None:
    args = parse_args()
    args.stop_token_ids = parse_stop_token_ids(args.stop_token_ids)
    args.blocked_token_ids = parse_stop_token_ids(args.blocked_token_ids)
    args.chat_template = None
    args.chat_template_sha256 = None
    if args.chat_template_model:
        args.chat_template, args.chat_template_sha256 = load_chat_template_text(
            args.chat_template_model,
            trust_remote_code=args.trust_remote_code,
        )
    args.merged_root.mkdir(parents=True, exist_ok=True)

    specs = discover_model_specs(args)
    if not specs:
        raise RuntimeError("No checkpoints to evaluate.")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    print(f"[setup] gpu_ids={gpu_ids}", flush=True)
    print(f"[setup] steps={[spec.step for spec in specs]}", flush=True)

    tasks = task_specs(args.data_dir, args.tasks)
    task_payloads = [
        {"name": task.name, "samples": load_samples(task.path, args.prompt_template)} for task in tasks
    ]
    for task in task_payloads:
        print(f"[setup] task {task['name']}: {len(task['samples'])} examples", flush=True)

    eval_run_config = build_eval_run_config(args, task_payloads, gpu_ids)
    prepare_eval_output_root(args.output_root, eval_run_config, replace=args.replace)

    length_tokenizer = load_length_tokenizer(args.length_tokenizer)
    wandb_run = init_wandb(args, specs)
    summary_rows: list[dict[str, Any]] = []

    for spec in specs:
        model_path = ensure_merged_model(spec, args.merged_root)
        output_paths = generate_for_model(model_path, spec.label, task_payloads, gpu_ids, args)

        task_metrics = {}
        for task_name, output_path in output_paths.items():
            metrics = grade_file(
                output_path,
                length_tokenizer,
                args.prompt_template,
                stop_token_ids=args.stop_token_ids,
                max_tokens=args.max_tokens,
            )
            task_metrics[task_name] = metrics
            row = {
                "step": spec.step,
                "model_label": spec.label,
                "task": task_name,
                **metrics,
            }
            summary_rows.append(row)

        write_summary(args.output_root, summary_rows)

        if wandb_run is not None:
            log_payload = {}
            for task_name, metrics in task_metrics.items():
                for metric_name, value in metrics.items():
                    log_payload[f"eval/{task_name}/{metric_name}"] = value
            if task_metrics:
                log_payload["eval/avg_mean_score"] = sum(m["mean_score"] for m in task_metrics.values()) / len(task_metrics)
                log_payload["eval/avg_best_score"] = sum(m["best_score"] for m in task_metrics.values()) / len(task_metrics)
            wandb_run.log(log_payload, step=spec.step)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
