#!/usr/bin/env python3
"""Render the paired before/after excerpts that contrast output format.

Vanilla OPD keeps generating after the answer is already stated; with the
semantic-EOS-class fix the same prompt yields a response that ends. Both sides
are rendered from the same prompt, and the fixed side is a correct response, so
the contrast on the page is the shape of the output rather than its content.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

SUBSTITUTIONS = {"✅": "[+]", "\U0001f6a8": "[!]", "’": "'", "→": "->", "÷": "/"}

# The vanilla side is the rollout already shown in the tail figure; only its
# opening is needed here, since the point is where it goes rather than the whole.
LAYOUT = {
    "answer_repetition": {"index": 1, "vanilla_open": 8, "vanilla_close": 4, "fixed_max_lines": 40},
    "self_correction_loop": {"index": 2, "vanilla_open": 8, "vanilla_close": 4, "fixed_max_lines": 40},
    "token_repetition": {"index": 3, "vanilla_open": 6, "vanilla_close": 3, "fixed_max_lines": 40},
}


def asciify(text: str) -> str:
    for source, target in SUBSTITUTIONS.items():
        text = text.replace(source, target)
    return text


def fit(text: str, width: int = 76, max_line: int = 160) -> str:
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
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--tail-trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tails = {json.loads(l)["label"]: json.loads(l) for l in args.tail_trajectories.open()}
    for line in args.comparison.open():
        record = json.loads(line)
        label = record["label"]
        layout = LAYOUT[label]
        stem = f"cmp_{layout['index']}_{label}"

        # Prompts are single long lines of prose, so they are word-wrapped.
        # The truncation in fit() is for degenerate symbol runs, not for text.
        prompt = "\n".join(
            textwrap.fill(line, 76) if line.strip() else ""
            for line in asciify(record["prompt"]).splitlines()
        )
        (args.output_dir / f"{stem}_prompt.txt").write_text(prompt + "\n")

        vanilla = tails[label]["response"]
        head_lines = tails[label]["response_head_through_first_correct_answer"].splitlines()
        tail_lines = [l for l in tails[label]["response_tail_after_first_correct_answer"].splitlines() if l.strip()]
        opening = head_lines[-layout["vanilla_open"] :]
        after = tail_lines[: layout["vanilla_close"]]
        vanilla_excerpt = (
            "[... earlier reasoning elided ...]\n\n"
            + "\n".join(opening)
            + "\n\n[... the response does not stop here; it continues to the end of the\n"
            + "     generation budget without ever emitting an end-of-sequence token ...]\n\n"
            + "\n".join(after)
            + "\n\n[... and so on to the token limit ...]"
        )
        (args.output_dir / f"{stem}_vanilla.txt").write_text(fit(asciify(vanilla_excerpt)) + "\n")

        fixed = record["paired_fix_response"].strip()
        fixed_lines = fixed.splitlines()
        if len(fixed_lines) > layout["fixed_max_lines"]:
            keep = layout["fixed_max_lines"] - 12
            fixed = "\n".join(fixed_lines[:keep]) + "\n\n[... middle elided ...]\n\n" + "\n".join(fixed_lines[-10:])
        (args.output_dir / f"{stem}_fixed.txt").write_text(fit(asciify(fixed)) + "\n")

        print(f"{stem}: prompt/vanilla/fixed written "
              f"(vanilla {len(vanilla)} chars -> fixed {len(record['paired_fix_response'])} chars)")


if __name__ == "__main__":
    main()
