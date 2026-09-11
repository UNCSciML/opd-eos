#!/usr/bin/env python3
"""Measure teacher response lengths on the prompt stream used by OPD training."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=3200)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=7168)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop-token-ids", default="151645,151643")
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_token_ids(value: str) -> list[int]:
    result: list[int] = []
    for item in parse_csv(value):
        token_id = int(item)
        if token_id not in result:
            result.append(token_id)
    if not result:
        raise ValueError("At least one stop token id is required")
    return result


def resolve_gpu_ids(value: str) -> list[str]:
    if value != "auto":
        gpu_ids = parse_csv(value)
    else:
        gpu_ids = parse_csv(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        if not gpu_ids:
            count = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or "0")
            gpu_ids = [str(index) for index in range(count)]
    if not gpu_ids:
        raise ValueError("No GPUs found; set --gpu-ids or CUDA_VISIBLE_DEVICES")
    return gpu_ids


def normalize_chat(chat: Any) -> list[dict[str, Any]]:
    if hasattr(chat, "tolist"):
        chat = chat.tolist()
    if isinstance(chat, tuple):
        chat = list(chat)
    if not isinstance(chat, list) or not chat:
        raise ValueError(f"Expected a non-empty chat list, got {type(chat).__name__}")
    return [dict(message) for message in chat]


def load_indexed_chats(input_parquet: Path, max_prompts: int) -> list[tuple[int, list[dict[str, Any]]]]:
    if max_prompts <= 0:
        raise ValueError("--max-prompts must be positive")
    dataframe = pd.read_parquet(input_parquet, columns=["prompt"])
    if len(dataframe) < max_prompts:
        raise ValueError(
            f"Requested {max_prompts} prompts but dataset only contains {len(dataframe)} rows"
        )
    return [
        (index, normalize_chat(dataframe.iloc[index]["prompt"]))
        for index in range(max_prompts)
    ]


def split_prompt_shards(
    indexed_chats: list[tuple[int, list[dict[str, Any]]]], num_workers: int
) -> list[list[tuple[int, list[dict[str, Any]]]]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    return [indexed_chats[worker_index::num_workers] for worker_index in range(num_workers)]


def generate_prompt_batch(
    *,
    indexed_chats: list[tuple[int, list[dict[str, Any]]]],
    llm: Any,
    tokenizer: Any,
    sampling_params_cls: Any,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    stop_token_ids: list[int],
    enable_thinking: bool,
) -> list[dict[str, Any]]:
    formatted_prompts = [
        tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for _, chat in indexed_chats
    ]
    sampling = sampling_params_cls(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    outputs = llm.generate(formatted_prompts, sampling, use_tqdm=True)
    if len(outputs) != len(indexed_chats):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} prompt outputs for {len(indexed_chats)} prompts"
        )

    rows: list[dict[str, Any]] = []
    for (prompt_index, _), request_output in zip(indexed_chats, outputs):
        if len(request_output.outputs) != n:
            raise RuntimeError(
                f"Prompt {prompt_index} returned {len(request_output.outputs)} completions; expected {n}"
            )
        for rollout_index, completion in enumerate(request_output.outputs):
            token_ids = [int(token_id) for token_id in completion.token_ids]
            stop_reason = completion.stop_reason
            if stop_reason is None and completion.finish_reason == "stop" and token_ids:
                if token_ids[-1] in stop_token_ids:
                    stop_reason = token_ids[-1]
            rows.append(
                {
                    "prompt_index": int(prompt_index),
                    "rollout_index": int(rollout_index),
                    "response_length": len(token_ids),
                    "finish_reason": str(completion.finish_reason or ""),
                    "stop_reason": "" if stop_reason is None else str(stop_reason),
                }
            )
    return rows


def _worker_generate(worker_args: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        gpu_id,
        model_path,
        indexed_chats,
        n,
        max_tokens,
        max_model_len,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        stop_token_ids,
        enable_thinking,
        gpu_memory_utilization,
        max_num_batched_tokens,
    ) = worker_args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM, SamplingParams

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError:
        destroy_distributed_environment = None
        destroy_model_parallel = None

    llm = None
    try:
        print(
            f"[gpu {gpu_id}] loading teacher; prompts={len(indexed_chats)} n={n}",
            flush=True,
        )
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=1024,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            seed=0,
        )
        tokenizer = llm.get_tokenizer()
        return generate_prompt_batch(
            indexed_chats=indexed_chats,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=SamplingParams,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
            enable_thinking=enable_thinking,
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


def summarize_rows(rows: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty generation result")
    lengths = np.asarray([row["response_length"] for row in rows], dtype=np.int64)
    finish_reason_counts = Counter(str(row["finish_reason"]) for row in rows)
    stop_reason_counts = Counter(
        str(row["stop_reason"]) for row in rows if str(row["stop_reason"])
    )
    return {
        "num_generations": len(rows),
        "num_prompts": len({int(row.get("prompt_index", index)) for index, row in enumerate(rows)}),
        "response_length_mean": float(lengths.mean()),
        "response_length_std": float(lengths.std()),
        "response_length_median": float(np.median(lengths)),
        "response_length_p95": float(np.quantile(lengths, 0.95)),
        "response_length_min": int(lengths.min()),
        "response_length_max": int(lengths.max()),
        "max_tokens": int(max_tokens),
        "length_capped_fraction": float(finish_reason_counts.get("length", 0) / len(rows)),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
    }


def write_results(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (int(row["prompt_index"]), int(row["rollout_index"])))
    np.savez_compressed(
        output_dir / "teacher_response_lengths.npz",
        prompt_indices=np.asarray([row["prompt_index"] for row in rows], dtype=np.int32),
        rollout_indices=np.asarray([row["rollout_index"] for row in rows], dtype=np.int16),
        response_lengths=np.asarray([row["response_length"] for row in rows], dtype=np.int32),
        finish_reasons=np.asarray([row["finish_reason"] for row in rows]),
        stop_reasons=np.asarray([row["stop_reason"] for row in rows]),
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    stop_token_ids = parse_token_ids(args.stop_token_ids)
    gpu_ids = resolve_gpu_ids(args.gpu_ids)
    indexed_chats = load_indexed_chats(args.input_parquet, args.max_prompts)
    prompt_shards = split_prompt_shards(indexed_chats, len(gpu_ids))

    expected_generations = args.max_prompts * args.n
    print(
        f"Teacher rollout: prompts={args.max_prompts} n={args.n} "
        f"generations={expected_generations} workers={len(gpu_ids)} TP=1",
        flush=True,
    )
    worker_args = [
        (
            gpu_id,
            args.model_path,
            prompt_shards[worker_index],
            args.n,
            args.max_tokens,
            args.max_model_len,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            stop_token_ids,
            args.enable_thinking,
            args.gpu_memory_utilization,
            args.max_num_batched_tokens,
        )
        for worker_index, gpu_id in enumerate(gpu_ids)
        if prompt_shards[worker_index]
    ]

    rows: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(worker_args), mp_context=context
    ) as executor:
        futures = [executor.submit(_worker_generate, item) for item in worker_args]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())

    if len(rows) != expected_generations:
        raise RuntimeError(f"Generated {len(rows)} rows; expected {expected_generations}")
    summary = summarize_rows(rows, max_tokens=args.max_tokens)
    summary.update(
        {
            "input_parquet": str(args.input_parquet.resolve()),
            "model_path": str(args.model_path.resolve()),
            "n": args.n,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "stop_token_ids": stop_token_ids,
            "enable_thinking": args.enable_thinking,
            "tensor_parallel_size": 1,
            "num_workers": len(worker_args),
        }
    )
    write_results(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
