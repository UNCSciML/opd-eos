#!/usr/bin/env python3
"""Measure the future term that current-token OPD drops.

OPD uses `token_reward_direct`, so the advantage at position t is the immediate
token reward

    r_t = log p_T(a_t | s_t) - log p_S(a_t | s_t)

A return-to-go objective would instead use  A_t = sum_{t' >= t} r_{t'}.  The
difference is the realised future divergence

    F_t = sum_{t' > t} r_{t'}                            (F_t <= 0 in expectation)

which attaches to CONTINUING and is absent at a terminal action, because the
trajectory ends there. Current-token OPD therefore over-credits continuation by
|F_t|, which is one candidate explanation for length inflation surviving an EOS
surface-form fix.

This script only measures. It changes no training code and feeds nothing back
into any objective. For one student checkpoint it

  1. generates on-policy rollouts with the training prompt stream and rollout
     settings,
  2. scores every generated token under the student and the teacher, and
  3. at every interior position of a TRUNCATED rollout, where the model did in
     fact continue, compares how the two objectives rank stopping against
     continuing:

        current-token  stop iff  r_eos > r_cont
        return-to-go   stop iff  r_eos > r_cont + F

     `flip_fraction` is the share of positions that the missing term alone
     would have turned into a stop. The theory predicts it is large while
     length inflates and collapses around the step where length turns over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_teacher_eos_scoring import build_prompt_ids, generate, terminal_token_ids  # noqa: E402

CHUNK = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-model", required=True, help="checkpoint to measure")
    p.add_argument("--teacher-model", required=True)
    p.add_argument("--chat-template-model", default=None,
                   help="defaults to the teacher, matching training")
    p.add_argument("--input-parquet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", default=None, help="tag for the output files")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=7168)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_model(model_path: str):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    # Resolve the architecture by name: the Auto* mappings do not cover every
    # multimodal config (Gemma3Config is not in AutoModelForVision2Seq).
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architecture = config.architectures[0]
    cls = getattr(transformers, architecture, None) or AutoModelForCausalLM
    model = cls.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    return model, cls.__name__


def score_sequences(model_path, prompt_ids, rows, eos_ids, label):
    """Per-position log-prob of the sampled token and of the terminal class.

    Returns two ragged lists aligned with `rows`, covering the completion only.
    Full log-softmax is never materialised: the normaliser is reduced in chunks,
    which keeps a 7k x 262k vocabulary within a few GB.
    """
    model, cls_name = _load_model(model_path)
    print(f"[{label}] scoring with {cls_name}", flush=True)
    eos_index = torch.tensor(eos_ids, device="cuda")

    lp_sampled, lp_terminal = [], []
    with torch.no_grad():
        for i, row in enumerate(rows):
            prompt = prompt_ids[row["prompt_index"]]
            ids = prompt + row["completion_ids"]
            x = torch.tensor([ids], device="cuda")
            logits = model(input_ids=x).logits[0]
            # Position j predicts token j+1, so the completion starts at
            # len(prompt)-1 and the last useful position is len(ids)-2.
            lo, hi = len(prompt) - 1, len(ids) - 1
            seg = logits[lo:hi]
            target = x[0, lo + 1 : hi + 1]
            a_parts, e_parts = [], []
            for s in range(0, seg.shape[0], CHUNK):
                block = seg[s : s + CHUNK].float()
                logz = torch.logsumexp(block, dim=-1)
                a_parts.append(block.gather(-1, target[s : s + CHUNK, None])[:, 0] - logz)
                e_parts.append(torch.logsumexp(block.index_select(-1, eos_index), dim=-1) - logz)
            lp_sampled.append(torch.cat(a_parts).cpu().numpy().astype(np.float32))
            lp_terminal.append(torch.cat(e_parts).cpu().numpy().astype(np.float32))
            del logits, seg
            if (i + 1) % 32 == 0:
                torch.cuda.empty_cache()
                print(f"[{label}] {i + 1}/{len(rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return lp_sampled, lp_terminal


def summarise(r_cont_list, r_eos_list, terminated) -> dict:
    """Compare the two objectives' stop-vs-continue ranking, position by position."""
    out = {}
    for group, keep in (("truncated", ~terminated), ("terminated", terminated),
                        ("all", np.ones_like(terminated))):
        flips, ratios, f_abs, r_abs, n_pos = [], [], [], [], 0
        for r_cont, r_eos, use in zip(r_cont_list, r_eos_list, keep):
            if not use or r_cont.size < 2:
                continue
            # F_t = sum_{t' > t} r_cont[t']
            future = np.concatenate([np.cumsum(r_cont[::-1])[::-1][1:], [0.0]])
            # Only interior positions: the model actually chose to continue there.
            f, rc, re = future[:-1], r_cont[:-1], r_eos[:-1]
            current_says_stop = re > rc
            rtg_says_stop = re > rc + f
            flips.append(np.mean(~current_says_stop & rtg_says_stop))
            f_abs.append(np.abs(f)); r_abs.append(np.abs(rc))
            ratios.append(np.abs(f) / np.maximum(np.abs(rc), 1e-6))
            n_pos += f.size
        if not flips:
            out[group] = {"n_sequences": 0, "n_positions": 0}
            continue
        f_all = np.concatenate(f_abs); r_all = np.concatenate(r_abs)
        out[group] = {
            "n_sequences": len(flips),
            "n_positions": int(n_pos),
            "flip_fraction_mean": float(np.mean(flips)),
            "flip_fraction_median": float(np.median(flips)),
            "abs_future_median": float(np.median(f_all)),
            "abs_future_mean": float(f_all.mean()),
            "abs_r_median": float(np.median(r_all)),
            "future_over_r_median": float(np.median(np.concatenate(ratios))),
        }
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.student_model).name
    template_model = args.chat_template_model or args.teacher_model

    frame = pd.read_parquet(args.input_parquet).head(args.num_prompts)
    _, prompt_ids = build_prompt_ids(args.student_model, template_model, frame["prompt"].tolist())

    student_eos = terminal_token_ids(args.student_model)
    teacher_eos = terminal_token_ids(args.teacher_model)
    scored_ids = sorted(set(student_eos) | set(teacher_eos))
    print(f"student terminal ids={student_eos}  teacher={teacher_eos}  scored={scored_ids}", flush=True)

    rows = generate(args.student_model, prompt_ids, student_eos, args, f"{label}/rollout")
    if not rows:
        raise SystemExit("no completions generated")

    s_a, s_e = score_sequences(args.student_model, prompt_ids, rows, scored_ids, f"{label}/student")
    t_a, t_e = score_sequences(args.teacher_model, prompt_ids, rows, scored_ids, f"{label}/teacher")

    r_cont = [tt - ss for tt, ss in zip(t_a, s_a)]   # what OPD uses as the advantage
    r_eos = [tt - ss for tt, ss in zip(t_e, s_e)]    # immediate reward of stopping here
    terminated = np.array([r["terminated"] for r in rows], dtype=bool)
    lengths = np.array([len(r["completion_ids"]) for r in rows], dtype=np.int64)

    offsets = np.concatenate([[0], np.cumsum([x.size for x in r_cont])]).astype(np.int64)
    np.savez_compressed(
        args.output_dir / f"{label}_future_term.npz",
        r_cont=np.concatenate(r_cont), r_eos=np.concatenate(r_eos),
        offsets=offsets, terminated=terminated, lengths=lengths,
        prompt_index=np.array([r["prompt_index"] for r in rows], dtype=np.int64),
        eos_token_ids=np.array(scored_ids, dtype=np.int64),
    )

    summary = {
        "label": label,
        "student_model": str(args.student_model),
        "teacher_model": str(args.teacher_model),
        "num_prompts": args.num_prompts, "n": args.n, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "n_completions": len(rows),
        "terminated_fraction": float(terminated.mean()),
        "length_mean": float(lengths.mean()), "length_median": float(np.median(lengths)),
        "groups": summarise(r_cont, r_eos, terminated),
    }
    (args.output_dir / f"{label}_future_term.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
