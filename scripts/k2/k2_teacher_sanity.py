#!/usr/bin/env python
"""Teacher (main) sanity check under the proposed non-thinking format + prefix set B builder.

Greedy HF generation (official impl) on AMC23 + AIME24 with the derived non-thinking template,
stop on {1, 250019}.  Reports termination rate, which stop token, boxed-format rate, accuracy,
lengths, and p(1)/p(250019)/p(STOP) at the terminal step (from generation scores).  Terminated
responses with a valid final answer become prefix set B (prompt + generated tokens minus the
stop token), frozen as token IDs.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PR / "scripts/val/eval"))
from utils import grade_answer_verl, extract_answer  # noqa: E402
EOS = [1, 250019]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tokenizer", required=True, help="llama-view dir (main tokenizer + derived template)")
    ap.add_argument("--impl", choices=["official", "patched"], default="official"); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--tasks", default="AMC23,AIME24"); ap.add_argument("--max-new-tokens", type=int, default=4096); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k2_equivalence import load_hf
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer); tok.padding_side = "left"; tok.pad_token_id = 1
    m = load_hf(a.impl, a.model, a.dtype, a.attn)
    items = []
    TTRL = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."  # == evaluator PROMPT_TEMPLATES["ttrl"]
    for task in a.tasks.split(","):
        df = pd.read_parquet(PR / f"scripts/val/data/{task}/test.parquet")  # same source as the checkpoint evaluator
        for i in range(len(df)):
            q = TTRL.format(problem=str(df.at[i, "prompt"][0]["content"]).strip())
            gold = str(df.at[i, "reward_model"]["ground_truth"]).strip()
            text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            items.append(dict(task=task, example_id=i, question=q, gold=str(gold), prompt=text, prompt_ids=tok.encode(text, add_special_tokens=False)))
    res = []; t0 = time.time()
    for b in range(0, len(items), a.batch):
        batch = items[b:b + a.batch]
        enc = tok([x["prompt"] for x in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens, eos_token_id=EOS, pad_token_id=1)
        seqs = g[:, enc.input_ids.shape[1]:]
        for j, x in enumerate(batch):
            ids = seqs[j].tolist()
            # strip padding after the first stop token
            stop_pos = next((k for k, t in enumerate(ids) if t in EOS), None)
            gen = ids[:stop_pos] if stop_pos is not None else [t for t in ids if t != 1]
            stop_tok = ids[stop_pos] if stop_pos is not None else None
            # terminal-step distribution: one extra forward on prompt+generated tokens (stop token excluded)
            with torch.no_grad():
                full = torch.tensor([x["prompt_ids"] + gen], device="cuda")
                sc = torch.log_softmax(m(input_ids=full, use_cache=False).logits[0, -1].float(), -1)
            resp = tok.decode(gen, skip_special_tokens=False)
            correct = bool(grade_answer_verl(resp, x["gold"], answer_format="auto")); boxed = extract_answer(resp, answer_format="auto") is not None
            res.append(dict(task=x["task"], example_id=x["example_id"], gold=x["gold"], n_gen=len(gen), terminated=stop_pos is not None, stop_token=stop_tok,
                            p_eod_at_end=float(sc[1].exp()), p_eot_at_end=float(sc[250019].exp()), boxed=boxed, correct=correct, response=resp, prompt_ids=x["prompt_ids"], gen_ids=gen))
        print(f"  {b+len(batch)}/{len(items)} done ({time.time()-t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "teacher_sanity_generations.json", "w"))
    term = [r for r in res if r["terminated"]]
    summ = dict(n=len(res), terminated_rate=len(term) / len(res), stop_eot_frac=float(np.mean([r["stop_token"] == 250019 for r in term])) if term else None, stop_eod_frac=float(np.mean([r["stop_token"] == 1 for r in term])) if term else None,
                boxed_rate=float(np.mean([r["boxed"] for r in res])), accuracy={t: float(np.mean([r["correct"] for r in res if r["task"] == t])) for t in a.tasks.split(",")},
                mean_len=float(np.mean([r["n_gen"] for r in res])), median_len=float(np.median([r["n_gen"] for r in res])),
                p_eot_at_end_mean=float(np.mean([r["p_eot_at_end"] for r in term])) if term else None, p_eod_at_end_mean=float(np.mean([r["p_eod_at_end"] for r in term])) if term else None,
                p_stop_at_end_mean=float(np.mean([r["p_eot_at_end"] + r["p_eod_at_end"] for r in term])) if term else None, max_new_tokens=a.max_new_tokens)
    json.dump(summ, open(out / "teacher_sanity_summary.json", "w"), indent=1); print("SUMMARY", json.dumps(summ))
    # prefix set B: terminated + boxed (valid final answer), stop token removed
    B = [dict(idx=k, task=r["task"], example_id=r["example_id"], gold=r["gold"], correct=r["correct"], stop_token=r["stop_token"], n_tokens=len(r["prompt_ids"]) + len(r["gen_ids"]), token_ids=r["prompt_ids"] + r["gen_ids"]) for k, r in enumerate([r for r in res if r["terminated"] and r["boxed"]])]
    json.dump(dict(description="prefix set B: K2-main greedy non-thinking solutions (terminated, boxed), stop token removed", n=len(B), prefixes=B), open(out / "prefix_set_B.json", "w"))
    print(f"prefix set B: n={len(B)} (terminated&boxed) ; lengths min/med/max = {min(p['n_tokens'] for p in B) if B else 0}/{int(np.median([p['n_tokens'] for p in B])) if B else 0}/{max(p['n_tokens'] for p in B) if B else 0}")


if __name__ == "__main__":
    main()
