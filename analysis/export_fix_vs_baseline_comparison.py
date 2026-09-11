#!/usr/bin/env python3
"""Compare vanilla OPD against the semantic-EOS-class fix on the same problems.

The tail figure shows three vanilla-OPD rollouts that never stop. This pairs each
with the same problem under the semantic_class fix, evaluated with the same
prompt template, budget, sampling settings and engine seed. Sample indices are
not matched across the two runs, because the 16 samples come from one seeded
stream per run and index i carries no correspondence between runs; the fix side
is represented by its shortest correct sample instead.

For every problem it reports all 16 samples per side, so the paired example is
visible as a draw from a distribution rather than as a single hand-picked case.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "val" / "eval"))
from utils import grade_answer_verl  # noqa: E402

CASES = [
    {"label": "answer_repetition", "step": 100, "dataset": "amc23", "example_id": 10, "truth": "5"},
    {"label": "self_correction_loop", "step": 100, "dataset": "aime24", "example_id": 7, "truth": "25"},
    {"label": "token_repetition", "step": 200, "dataset": "amc23", "example_id": 0, "truth": "27"},
]


def load_side(root: Path, step: int, dataset: str, example_id: int, truth: str) -> list[dict]:
    path = Path(sorted(glob.glob(str(root / f"step_{step:04d}" / f"{dataset}_*.jsonl")))[0])
    rows = [json.loads(line) for line in path.open()]
    with np.load(path.with_name(path.stem + ".tokens.npz")) as bundle:
        offsets = bundle["offsets"]
        lengths = np.diff(offsets)
    out = []
    for index, row in enumerate(rows):
        if row["example_id"] != example_id:
            continue
        response = row["response"]
        out.append(
            {
                "rollout_index": row["seed"],
                "correct": bool(grade_answer_verl(response, truth)),
                "tokens": int(lengths[index]),
                "chars": len(response),
                "zlib_ratio": round(len(zlib.compress(response.encode(), 9)) / max(len(response.encode()), 1), 5),
                "response": response,
                "prompt": row["prompt"],
            }
        )
    return sorted(out, key=lambda d: d["rollout_index"])


def describe(samples: list[dict], budget: int) -> dict:
    tokens = np.array([s["tokens"] for s in samples])
    return {
        "n_samples": len(samples),
        "n_correct": int(sum(s["correct"] for s in samples)),
        "tokens_median": int(np.median(tokens)),
        "tokens_max": int(tokens.max()),
        "tokens_min": int(tokens.min()),
        "n_hitting_budget": int((tokens >= budget).sum()),
        "zlib_ratio_median": round(float(np.median([s["zlib_ratio"] for s in samples])), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--fix-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records, summaries = [], []
    for case in CASES:
        base = load_side(args.baseline_root, case["step"], case["dataset"], case["example_id"], case["truth"])
        fix = load_side(args.fix_root, case["step"], case["dataset"], case["example_id"], case["truth"])
        if base[0]["prompt"].strip() != fix[0]["prompt"].strip():
            raise ValueError(f"prompt mismatch for {case['label']}; the two sides are not the same problem")

        correct_fix = sorted([s for s in fix if s["correct"]], key=lambda s: s["tokens"])
        if not correct_fix:
            raise ValueError(f"no correct sample under the fix for {case['label']}")
        chosen = correct_fix[0]

        summaries.append(
            {
                **{k: case[k] for k in ("label", "step", "dataset", "example_id", "truth")},
                "baseline": describe(base, args.budget),
                "semantic_class": describe(fix, args.budget),
                "paired_fix_sample": {
                    "rollout_index": chosen["rollout_index"],
                    "tokens": chosen["tokens"],
                    "zlib_ratio": chosen["zlib_ratio"],
                },
            }
        )
        records.append(
            {
                **{k: case[k] for k in ("label", "step", "dataset", "example_id", "truth")},
                "prompt": base[0]["prompt"],
                "baseline_samples": [{k: v for k, v in s.items() if k != "prompt"} for s in base],
                "semantic_class_samples": [{k: v for k, v in s.items() if k != "prompt"} for s in fix],
                "paired_fix_response": chosen["response"],
            }
        )

    with args.output_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output_summary.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))

    print(f"{'case':<22}{'side':<16}{'correct':>9}{'tok med':>9}{'tok max':>9}{'at budget':>11}{'zlib med':>10}")
    print("-" * 86)
    for s in summaries:
        for side in ("baseline", "semantic_class"):
            d = s[side]
            print(f"{s['label'] if side=='baseline' else '':<22}{side:<16}"
                  f"{d['n_correct']:>6}/16{d['tokens_median']:>9}{d['tokens_max']:>9}"
                  f"{d['n_hitting_budget']:>8}/16{d['zlib_ratio_median']:>10.3f}")
        print()


if __name__ == "__main__":
    main()
