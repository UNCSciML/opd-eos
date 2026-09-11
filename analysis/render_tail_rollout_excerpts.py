#!/usr/bin/env python3
"""Render the appendix excerpt files for the three post-answer tail rollouts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Non-ASCII characters that survive into the excerpts, mapped to ASCII stand-ins.
SUBSTITUTIONS = {"✅": "[+]", "\U0001f6a8": "[!]", "’": "'", "→": "->", "÷": "/"}

# The elided motif is named per example. The most frequent tail *line* is not it:
# for example 1 that is the "---" separator, and for example 3 it is "$$".
SPEC = {
    "answer_repetition": {
        "index": 1, "head_lines": 22, "tail_open": 10, "tail_close": 6,
        "motif": r"\boxed{5}", "count_key": "boxed_repeats",
    },
    "self_correction_loop": {
        "index": 2, "head_lines": 20, "tail_open": 12, "tail_close": 6,
        "motif": "the restart line", "count_key": "restart_repeats",
    },
    "token_repetition": {
        "index": 3, "head_lines": 18, "tail_open": 6, "tail_close": 4,
        "motif": "[+]", "count_key": "symbol_repeats",
    },
}


def asciify(text: str) -> str:
    for source, target in SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text


def wrap_symbol_runs(text: str, width: int = 76, max_line: int = 160) -> str:
    """Truncate overlong lines, then hard-wrap, so excerpts fit the column.

    Truncation has to come first. A degenerate line can be thousands of
    characters of a single repeated symbol, and wrapping that alone produces
    hundreds of output lines.
    """
    out = []
    for line in text.splitlines():
        if len(line) > max_line:
            dropped = len(line) - width * 2
            line = line[: width * 2] + f" [... {dropped} more characters on this line elided ...]"
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = {s["label"]: s for s in json.loads(args.stats.read_text())}
    rows = [json.loads(line) for line in args.trajectories.open()]

    for row in rows:
        label = row["label"]
        spec = SPEC[label]
        stem = f"traj_{spec['index']}_{label}"
        stat = stats[label]

        (args.output_dir / f"{stem}_prompt.txt").write_text(
            wrap_symbol_runs(asciify(row["prompt"])) + "\n"
        )

        head_lines = row["response_head_through_first_correct_answer"].splitlines()
        head = "[... earlier reasoning elided ...]\n\n" + "\n".join(head_lines[-spec["head_lines"] :])
        (args.output_dir / f"{stem}_head.txt").write_text(wrap_symbol_runs(asciify(head)) + "\n")

        tail_lines = [l for l in row["response_tail_after_first_correct_answer"].splitlines()]
        while tail_lines and not tail_lines[0].strip():
            tail_lines.pop(0)
        opening = tail_lines[: spec["tail_open"]]
        closing = tail_lines[-spec["tail_close"] :]
        tail_text = row["response_tail_after_first_correct_answer"]
        if label == "answer_repetition":
            count = tail_text.count(r"\boxed{5}")
        elif label == "self_correction_loop":
            # Count complete occurrences; the budget cuts the last one short.
            count = len(re.findall(r"Let’s try \$x = 2\$, \$y = 12\.5\$", tail_text))
        else:
            count = dict(stat["tail_most_common_symbols"])["✅"]
        elision = (
            f"[... continues this way; {spec['motif']} occurs {count} times "
            f"in the continuation, middle elided ...]"
        )
        tail = "\n".join(opening) + f"\n\n{elision}\n\n" + "\n".join(closing)
        (args.output_dir / f"{stem}_tail.txt").write_text(wrap_symbol_runs(asciify(tail)) + "\n")
        print(f"{stem}: prompt/head/tail written; elision = {elision}")


if __name__ == "__main__":
    main()
