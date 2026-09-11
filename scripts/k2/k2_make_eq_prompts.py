#!/usr/bin/env python
"""Fixed prompt set for the K2 implementation-equivalence test (token IDs frozen once)."""
import argparse, json, random
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--prefix-set", required=True)
    ap.add_argument("--n-ordinary", type=int, default=8); ap.add_argument("--n-terminal", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=3072); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    random.seed(0); rows = []
    TTRL = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
    for task in ["AMC23", "AIME24"]:
        df = pd.read_parquet(f"scripts/val/data/{task}/test.parquet")
        for i in random.sample(range(len(df)), a.n_ordinary // 2):
            q = TTRL.format(problem=str(df.at[i, "prompt"][0]["content"]).strip())
            text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            rows.append(dict(name=f"ordinary_{task}_{i}", kind="ordinary", text=text, token_ids=tok.encode(text, add_special_tokens=False)))
    pre = json.load(open(a.prefix_set))["prefixes"]
    pre = sorted([p for p in pre if p["n_tokens"] <= a.max_len], key=lambda p: p["n_tokens"])
    picks = [pre[int(round(k * (len(pre) - 1) / max(a.n_terminal - 1, 1)))] for k in range(a.n_terminal)]  # spread over lengths
    for p in picks:
        rows.append(dict(name=f"terminal_A{p['idx']}_{p['task']}", kind="terminal", text=p["text"], token_ids=p["token_ids"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(rows, open(a.out, "w"))
    print(f"wrote {a.out}: {len(rows)} prompts; lengths = {[len(r['token_ids']) for r in rows]}")


if __name__ == "__main__":
    main()
