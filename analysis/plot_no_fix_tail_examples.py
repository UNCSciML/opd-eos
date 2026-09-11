#!/usr/bin/env python3
"""Draw representative post-answer degeneration from no-EOS-fix rollouts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge


DEFAULT_EVAL_ROOT = Path(
    "eval_outputs/"
    "opd_ttrl_baseline_sampled_clean_bs16_n4_200step_20260901_"
    "ttrl_baseline_think-false_m8192_n16_4gpu"
)


@dataclass(frozen=True)
class ExampleSpec:
    title: str
    step: int
    dataset: str
    example_id: int
    seed: int
    expected_answer: str
    behavior: str
    motif: str = ""


@dataclass(frozen=True)
class Example:
    spec: ExampleSpec
    answer: str
    context: str
    token_count: int
    repetition_count: int
    alert_count: int = 0
    check_count: int = 0


EXAMPLE_SPECS = (
    ExampleSpec(
        title="Repeated final answer",
        step=100,
        dataset="amc23",
        example_id=10,
        seed=8,
        expected_answer="5",
        behavior="answer",
    ),
    ExampleSpec(
        title="Repeated reasoning",
        step=100,
        dataset="aime24",
        example_id=7,
        seed=1,
        expected_answer="25",
        behavior="reasoning",
        motif="Let’s try $x = 2$, $y = 12.5$.",
    ),
    ExampleSpec(
        title="Repeated emoji",
        step=200,
        dataset="amc23",
        example_id=0,
        seed=13,
        expected_answer="27",
        behavior="emoji",
    ),
)


def boxed_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    marker = r"\boxed{"
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


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith(r"\boxed{") and answer.endswith("}"):
        answer = answer[len(r"\boxed{") : -1]
    return re.sub(r"\s+", "", answer)


def clean_context(prefix: str) -> str:
    candidates = []
    for line in prefix.splitlines():
        line = line.strip()
        if not line or line in {"---", "$$", r"\[", r"\]"} or line.startswith("###"):
            continue
        line = line.replace("**", "").replace("$", "")
        line = line.replace(r"\cdot", "×").replace(r"\text", "")
        candidates.append(line)
    context = candidates[-1] if candidates else "The model completes its solution."
    if len(context) > 92:
        context = "…" + context[-91:]
    return context


def _load_token_count(jsonl_path: Path, row_index: int) -> int:
    token_path = jsonl_path.with_name(jsonl_path.stem + ".tokens.npz")
    with np.load(token_path) as archive:
        offsets = archive["offsets"]
        return int(offsets[row_index + 1] - offsets[row_index])


def load_example(eval_root: Path, spec: ExampleSpec) -> Example:
    step_dir = eval_root / f"step_{spec.step:04d}"
    matches = sorted(step_dir.glob(f"{spec.dataset}_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {spec.dataset} JSONL in {step_dir}, found {len(matches)}")
    jsonl_path = matches[0]
    rows = [json.loads(line) for line in jsonl_path.open()]
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if row["example_id"] == spec.example_id and row["seed"] == spec.seed
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one example_id={spec.example_id}, seed={spec.seed} in {jsonl_path}"
        )
    row_index, row = selected[0]
    spans = boxed_spans(row["response"])
    if not spans:
        raise ValueError(f"Selected response in {jsonl_path} has no complete boxed answer")
    first_start, first_end, first_answer = spans[0]
    if normalize_answer(first_answer) != normalize_answer(spec.expected_answer):
        raise ValueError(
            f"First boxed answer is {first_answer!r}, expected {spec.expected_answer!r}"
        )
    tail = row["response"][first_end:]
    correct_boxes_after = sum(
        normalize_answer(content) == normalize_answer(spec.expected_answer)
        for _, _, content in spans[1:]
    )
    if spec.behavior == "answer":
        repetition_count = correct_boxes_after
    elif spec.behavior == "reasoning":
        repetition_count = tail.count(spec.motif)
    elif spec.behavior == "emoji":
        repetition_count = sum(tail.count(character) for character in ("🚨", "✅"))
    else:
        raise ValueError(f"Unknown behavior: {spec.behavior}")
    return Example(
        spec=spec,
        answer=first_answer,
        context=clean_context(row["response"][:first_start]),
        token_count=_load_token_count(jsonl_path, row_index),
        repetition_count=repetition_count,
        alert_count=tail.count("🚨"),
        check_count=tail.count("✅"),
    )


def _rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float, **kwargs):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=ax.transAxes,
        **kwargs,
    )
    ax.add_patch(box)
    return box


def _draw_check(ax: plt.Axes, x: float, y: float, size: float) -> None:
    _rounded_box(
        ax,
        x,
        y,
        size,
        size,
        facecolor="#2A9D6F",
        edgecolor="none",
        zorder=4,
    )
    ax.plot(
        [x + 0.22 * size, x + 0.43 * size, x + 0.80 * size],
        [y + 0.50 * size, y + 0.27 * size, y + 0.73 * size],
        color="white",
        linewidth=1.5,
        solid_capstyle="round",
        transform=ax.transAxes,
        zorder=5,
    )


def _draw_alert(ax: plt.Axes, x: float, y: float, size: float) -> None:
    ax.add_patch(
        Wedge(
            (x + 0.5 * size, y + 0.42 * size),
            0.38 * size,
            0,
            180,
            facecolor="#D95F59",
            edgecolor="#A53B36",
            linewidth=0.6,
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.add_patch(
        Rectangle(
            (x + 0.12 * size, y + 0.34 * size),
            0.76 * size,
            0.14 * size,
            facecolor="#555555",
            edgecolor="none",
            transform=ax.transAxes,
            zorder=4,
        )
    )
    ax.plot(
        [x + 0.5 * size, x + 0.5 * size],
        [y + 0.53 * size, y + 0.70 * size],
        color="white",
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=5,
    )


def _tail_text(example: Example) -> str:
    if example.spec.behavior == "answer":
        return (
            f"### Final Answer:\n\\boxed{{{example.answer}}}\n"
            f"          ...\n\\boxed{{{example.answer}}}\n"
            f"repeated {example.repetition_count:,} more times"
        )
    if example.spec.behavior == "reasoning":
        motif = example.spec.motif.replace("$", "")
        return f"{motif}\n{motif}\n          ...\nrepeated {example.repetition_count:,} times"
    return ""


def build_figure(examples: list[Example]) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.parse_math": False,
        }
    )
    figure = plt.figure(figsize=(10.2, 5.8), facecolor="white")
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "No EOS alignment: the model answers, then keeps generating",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.914,
        "Representative no-fix TTRL evaluation rollouts; repetitions are collapsed",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.text(0.245, 0.865, "VALID RESPONSE", ha="center", fontsize=9, color="#237A57")
    ax.text(0.742, 0.865, "DEGENERATE CONTINUATION", ha="center", fontsize=9, color="#A64B42")

    row_bottoms = (0.62, 0.36, 0.10)
    for row_index, (example, bottom) in enumerate(zip(examples, row_bottoms), start=1):
        height = 0.205
        ax.text(
            0.035,
            bottom + height - 0.005,
            f"{row_index}. {example.spec.title}",
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.text(
            0.035,
            bottom + height - 0.042,
            (
                f"{example.spec.dataset.upper()} · step {example.spec.step} · "
                f"example {example.spec.example_id} · seed {example.spec.seed}"
            ),
            ha="left",
            va="top",
            fontsize=7.2,
            color="#666666",
        )

        left_x, left_w = 0.275, 0.235
        right_x, right_w = 0.555, 0.405
        _rounded_box(
            ax,
            left_x,
            bottom,
            left_w,
            height,
            facecolor="#EDF8F2",
            edgecolor="#73B79D",
            linewidth=1.0,
        )
        _rounded_box(
            ax,
            right_x,
            bottom,
            right_w,
            height,
            facecolor="#FFF1ED",
            edgecolor="#DF9488",
            linewidth=1.0,
        )

        ax.text(
            left_x + 0.015,
            bottom + height - 0.035,
            example.context,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            wrap=True,
        )
        ax.text(
            left_x + left_w / 2,
            bottom + 0.047,
            rf"\boxed{{{example.answer}}}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="monospace",
            color="#185C42",
            fontweight="bold",
        )

        boundary_x = 0.533
        ax.plot(
            [boundary_x, boundary_x],
            [bottom + 0.005, bottom + height - 0.005],
            transform=ax.transAxes,
            color="#B5483F",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            boundary_x,
            bottom + height + 0.012,
            "Expected stop",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#A33B33",
        )
        ax.add_patch(
            FancyArrowPatch(
                (left_x + left_w + 0.006, bottom + height / 2),
                (right_x - 0.008, bottom + height / 2),
                arrowstyle="-|>",
                mutation_scale=8,
                color="#B5483F",
                linewidth=0.9,
                transform=ax.transAxes,
            )
        )

        if example.spec.behavior == "emoji":
            icon_y = bottom + 0.093
            icon_size = 0.030
            icon_x = right_x + 0.035
            for _ in range(3):
                _draw_alert(ax, icon_x, icon_y, icon_size)
                icon_x += 0.042
            ax.text(icon_x + 0.002, icon_y + 0.014, "→", transform=ax.transAxes, va="center")
            icon_x += 0.032
            for _ in range(4):
                _draw_check(ax, icon_x, icon_y, icon_size)
                icon_x += 0.042
            ax.text(icon_x + 0.001, icon_y + 0.014, "…", transform=ax.transAxes, va="center")
            ax.text(
                right_x + 0.025,
                bottom + 0.048,
                f"U+1F6A8 ALERT × {example.alert_count:,}     U+2705 CHECK × {example.check_count:,}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7.7,
                fontfamily="monospace",
            )
        else:
            ax.text(
                right_x + 0.025,
                bottom + height / 2,
                _tail_text(example),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=7.5,
                linespacing=1.35,
                fontfamily="monospace",
            )

        ax.text(
            right_x + right_w - 0.012,
            bottom + 0.018,
            f"→ {example.token_count}-token cap",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#A33B33",
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.035,
        "The boxed answer is already correct in all three examples; only the post-answer tail is abbreviated.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    examples = [load_example(args.eval_root, spec) for spec in EXAMPLE_SPECS]
    figure = build_figure(examples)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    figure.savefig(
        args.output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(figure)
    print(f"Wrote representative no-fix tail examples to {args.output_prefix}.{{png,pdf}}")


if __name__ == "__main__":
    main()
