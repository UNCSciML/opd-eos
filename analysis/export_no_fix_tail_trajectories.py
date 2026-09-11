#!/usr/bin/env python3
"""Export the full trajectories behind the post-answer tail figure, and test the
claim that everything after the first correct answer is pure repetition.

The figure (analysis/plot_no_fix_tail_examples.py) shows only a trimmed excerpt.
This writes the complete prompt and response for each example to JSONL, then
measures the tail rather than asserting anything about it:

  - where the first correct boxed answer ends, in characters and in tokens;
  - what fraction of the tail's lines are ones already seen before it;
  - the zlib compression ratio of the tail, as a crude entropy proxy;
  - the shortest repeating period that reconstructs the tail, if one exists;
  - how many characters of the tail are NOT covered by that repeating period,
    which is the number that would falsify "it just repeats".

Note on the "seed" field of the eval JSONL: it holds the rollout index, not an
RNG seed. This eval ran with a single engine seed of 0 and batched n=16, so a
row's "seed" is which of the 16 samples it is. It is re-exported here as
"rollout_index" to stop that name from misleading anyone.
"""

from __future__ import annotations

import argparse
import collections
import json
import zlib
from pathlib import Path

import numpy as np

EXAMPLES = [
    {"label": "answer_repetition", "step": 100, "dataset": "amc23", "example_id": 10, "seed": 8, "expected": "5"},
    {"label": "self_correction_loop", "step": 100, "dataset": "aime24", "example_id": 7, "seed": 1, "expected": "25"},
    {"label": "token_repetition", "step": 200, "dataset": "amc23", "example_id": 0, "seed": 13, "expected": "27"},
]


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans, cursor, marker = [], 0, r"\boxed{"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            return spans
        depth = 0
        for index in range(start + len(r"\boxed"), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1, text[start + len(marker) : index]))
                    cursor = index + 1
                    break
        else:
            return spans


def first_covering_token(tokenizer, token_ids: list[int], char_offset: int) -> int:
    """Smallest token count whose decoding already covers `char_offset` characters."""
    low, high = 0, len(token_ids)
    while low < high:
        mid = (low + high) // 2
        if len(tokenizer.decode(token_ids[:mid], skip_special_tokens=True)) >= char_offset:
            high = mid
        else:
            low = mid + 1
    return low


def dominant_period(text: str, max_period: int = 4000, threshold: float = 0.95) -> dict:
    """Smallest shift p at which the text agrees with itself at least `threshold`.

    Exact periodicity is the wrong test here: a generated tail starts mid-motif
    and is cut off mid-motif by the token limit, so a single mismatched character
    at either end defeats it. Approximate self-agreement measures what actually
    matters, namely what fraction of the tail is predictable from p characters
    earlier, and how many characters are not.
    """
    codes = np.array([ord(c) for c in text], dtype=np.int32)
    n = codes.size
    limit = min(max_period, n // 3)
    best = {"period": None, "match_rate": 0.0}
    for period in range(1, max(limit, 1) + 1):
        rate = float((codes[period:] == codes[:-period]).mean())
        if rate > best["match_rate"]:
            best = {"period": period, "match_rate": rate}
        if rate >= threshold:
            best = {"period": period, "match_rate": rate}
            break
    if best["period"] is None:
        return {"period": None, "match_rate": 0.0, "unit": "", "mismatched_chars": n}
    period = best["period"]
    mismatches = int((codes[period:] != codes[:-period]).sum())
    return {
        "period": period,
        "match_rate": round(best["match_rate"], 5),
        "unit": text[:period],
        "mismatched_chars": mismatches,
    }


def analyse_tail(response: str, answer_end: int) -> dict:
    head, tail = response[:answer_end], response[answer_end:]
    head_lines = {line.strip() for line in head.splitlines() if line.strip()}
    tail_lines = [line.strip() for line in tail.splitlines() if line.strip()]
    seen_before = sum(1 for line in tail_lines if line in head_lines)
    unique_tail = len(set(tail_lines))

    stripped = tail.strip()
    # Skip a short run-in so the measurement starts inside the motif rather than
    # on the transition into it.
    run_in = min(200, len(stripped) // 10)
    periodic = dominant_period(stripped[run_in:])

    symbols = collections.Counter(c for c in tail if ord(c) > 0x2000)
    return {
        "tail_chars": len(tail),
        "tail_most_common_lines": collections.Counter(tail_lines).most_common(3),
        "tail_most_common_symbols": symbols.most_common(3),
        "tail_nonblank_lines": len(tail_lines),
        "tail_unique_nonblank_lines": unique_tail,
        "tail_lines_already_seen_in_head": seen_before,
        "tail_zlib_ratio": round(len(zlib.compress(tail.encode(), 9)) / max(len(tail.encode()), 1), 5),
        "head_zlib_ratio": round(len(zlib.compress(head.encode(), 9)) / max(len(head.encode()), 1), 5),
        "repeat_period_chars": periodic["period"],
        "repeat_self_agreement": periodic["match_rate"],
        "repeating_unit": periodic["unit"][:300],
        "tail_chars_not_predicted_by_period": periodic["mismatched_chars"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records, summaries = [], []

    for spec in EXAMPLES:
        step_dir = args.eval_root / f"step_{spec['step']:04d}"
        jsonl_path = sorted(step_dir.glob(f"{spec['dataset']}_*.jsonl"))[0]
        rows = [json.loads(line) for line in jsonl_path.open()]
        hits = [
            (i, r) for i, r in enumerate(rows)
            if r["example_id"] == spec["example_id"] and r["seed"] == spec["seed"]
        ]
        if len(hits) != 1:
            raise ValueError(f"expected exactly one row for {spec}, found {len(hits)}")
        row_index, row = hits[0]
        response = row["response"]

        spans = [s for s in boxed_spans(response) if s[2].strip() == spec["expected"]]
        if not spans:
            raise ValueError(f"no boxed {spec['expected']!r} in {spec['label']}")
        first_start, first_end, _ = spans[0]

        token_path = jsonl_path.with_name(jsonl_path.stem + ".tokens.npz")
        with np.load(token_path) as bundle:
            offsets = bundle["offsets"]
            token_ids = bundle["token_ids"][offsets[row_index] : offsets[row_index + 1]]
        total_tokens = int(len(token_ids))
        # Exact character offset -> token index. A proportional estimate is wrong
        # here because the tail's characters-per-token ratio differs sharply from
        # the head's (emoji and symbol runs are token-dense).
        answer_token_estimate = first_covering_token(tokenizer, [int(t) for t in token_ids], first_end)

        stats = analyse_tail(response, first_end)
        stats.update(
            {
                "label": spec["label"],
                "step": spec["step"],
                "dataset": spec["dataset"],
                "example_id": spec["example_id"],
                "rollout_index": spec["seed"],
                "eval_engine_seed": 0,
                "expected_answer": spec["expected"],
                "total_response_chars": len(response),
                "total_response_tokens": total_tokens,
                "first_correct_answer_end_char": first_end,
                "first_correct_answer_end_token": answer_token_estimate,
                "redundant_tokens": total_tokens - answer_token_estimate,
                "redundant_fraction": round((total_tokens - answer_token_estimate) / max(total_tokens, 1), 4),
                "boxed_occurrences_total": len(boxed_spans(response)),
                "boxed_occurrences_with_expected_answer": len(spans),
                "source_jsonl": str(jsonl_path),
                "source_row_index": row_index,
            }
        )
        summaries.append(stats)
        records.append(
            {
                "label": spec["label"],
                "step": spec["step"],
                "dataset": spec["dataset"],
                "example_id": spec["example_id"],
                # The eval JSONL calls this "seed", but build_generation_row stores
                # the rollout index, not an RNG seed. This run used a single
                # engine seed of 0 with batched n=16, so these are samples 0..15
                # drawn from one seeded stream at temperature 0.7. Reproducing one
                # sample means rerunning the whole n=16 batch at seed 0 and taking
                # this index; rerunning with "seed" as the seed reproduces nothing.
                "rollout_index": spec["seed"],
                "eval_engine_seed": 0,
                "expected_answer": spec["expected"],
                "prompt": row["prompt"],
                "response": response,
                "response_head_through_first_correct_answer": response[:first_end],
                "response_tail_after_first_correct_answer": response[first_end:],
                "total_response_tokens": total_tokens,
                "token_ids": [int(t) for t in token_ids],
                "source_jsonl": str(jsonl_path),
                "source_row_index": row_index,
            }
        )

    with args.output_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output_summary.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
