#!/usr/bin/env python
"""Prefix set A: model-independent completed solutions as frozen K2 token-ID prefixes.

Source: correct responses from existing Qwen TTRL eval outputs (any checkpoint dirs given).
Each prefix = K2 non-thinking chat format of [user: TTRL question, assistant: solution cut
right after the final \\boxed{...}] with the closing <|ifm|im_end|> REMOVED, tokenised once
with the common (main) tokenizer view.  The next-token distribution at the end of the
prefix is where p(EOD=1) / p(EOT=250019) are probed.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

PR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PR / "scripts/val/eval")); sys.path.insert(0, str(PR / "scripts/analysis"))
from utils import grade_answer_verl  # noqa: E402
from symmetric_analysis import find_boxes  # noqa: E402


def cut_after_final_box(resp):
    boxes = list(find_boxes(resp))
    if not boxes:
        return None
    s, e, _ = boxes[-1]
    tail = resp[e:e + 4]
    # keep an immediately following closing "$", "$$", "." or "**" so the sentence is complete
    extra = 0
    for cand in ("$$.", "$$", "$.", "**.", "**", ".", "$"):
        if tail.startswith(cand):
            extra = len(cand); break
    return resp[:e + extra]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="eval output dir (with step_XXXX/*.jsonl)")
    ap.add_argument("--steps", default="0200,0180,0160")
    ap.add_argument("--tokenizer", required=True, help="K2 llama-view dir (main tokenizer + derived template)")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-question", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cands = {}
    for src in a.src:
        for st in a.steps.split(","):
            for f in sorted(Path(src).glob(f"step_{st}/*.jsonl")):
                task = f.name.split("_")[0]
                for l in open(f):
                    r = json.loads(l); q = r["prompt"]; gold = str(r["answer"]); resp = str(r["response"])
                    if not grade_answer_verl(resp, gold, answer_format="auto"):
                        continue
                    cut = cut_after_final_box(resp)
                    if cut is None or len(cut) < 40:
                        continue
                    key = (task, int(r["example_id"]))
                    cands.setdefault(key, []).append(dict(task=task, example_id=int(r["example_id"]), question=q, gold=gold, solution=cut, src=str(f)))
    random.seed(a.seed)
    keys = sorted(cands); random.shuffle(keys)
    out = []; end_marker = "<|ifm|im_end|>"
    for key in keys:
      pool = cands[key][:]; random.shuffle(pool); seen = set()
      for c in pool:
        if len(seen) >= a.per_question or c["solution"] in seen: continue
        seen.add(c["solution"])
        msgs = [{"role": "user", "content": c["question"]}, {"role": "assistant", "content": c["solution"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        assert text.endswith(end_marker), text[-40:]
        text = text[:-len(end_marker)]
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > a.max_tokens:
            continue
        out.append(dict(idx=len(out), task=c["task"], example_id=c["example_id"], gold=c["gold"], n_tokens=len(ids), text=text, token_ids=ids, source=c["src"]))
      if len(out) >= a.n:
        break
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    json.dump(dict(description=__doc__, tokenizer=a.tokenizer, template_sha256=__import__("hashlib").sha256(tok.chat_template.encode()).hexdigest(), n=len(out), prefixes=out), open(o / "prefix_set_A.json", "w"), indent=1)
    flat = np.concatenate([np.asarray(p["token_ids"], dtype=np.int32) for p in out]); offs = np.cumsum([0] + [p["n_tokens"] for p in out])
    np.savez(o / "prefix_set_A.tokens.npz", token_ids=flat, offsets=offs)
    lens = [p["n_tokens"] for p in out]
    print(f"prefix set A: n={len(out)} from {len(cands)} unique solved questions; tokens min/median/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}; tasks={ {t: sum(p['task']==t for p in out) for t in set(p['task'] for p in out)} }")
    print("example prefix tail:", repr(out[0]["text"][-160:])); print("tail ids:", out[0]["token_ids"][-6:])


if __name__ == "__main__":
    main()
