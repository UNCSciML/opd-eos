#!/usr/bin/env python
"""Measure a K2-Horizon model's response length / clip rate on a set of prompts with vLLM (Llama-view + plugin).

Example (teacher = main, training-like sampling, non-thinking template):
  python scripts/k2/k2_teacher_length_vllm.py --model $K2/K2-Horizon-7B-main-llamaview \
      --prompts train_prompts.jsonl --field prompt --n 4 --max-tokens 7168 --max-model-len 8192 \
      --temperature 1.0 --top-p 1.0 --out teacher_main_len.json
`--prompts` is a .jsonl (one object per line, text under --field) or a .json list of strings / objects.
Prompts are wrapped with the chat template (enable_thinking=False unless --thinking) unless --raw.
"""
import argparse, json, statistics
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

EOS_IDS = [1, 250019]  # <|ifm|endoftext|> (EOD), <|ifm|im_end|> (EOT); == generation_config.eos_token_id
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--field", default="prompt")
ap.add_argument("--n", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=7168); ap.add_argument("--max-model-len", type=int, default=8192)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--thinking", action="store_true"); ap.add_argument("--raw", action="store_true"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--out", required=True)
a = ap.parse_args()

if a.prompts.endswith(".jsonl"):
    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
else:
    items = json.load(open(a.prompts))
texts = [(it if isinstance(it, str) else it[a.field]) for it in items]
if a.limit: texts = texts[: a.limit]
tok = AutoTokenizer.from_pretrained(a.model)
if a.raw:
    prompts = texts
else:
    kw = {} if a.thinking else {"enable_thinking": False}
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, **kw) for t in texts]
llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, tensor_parallel_size=a.tp,
          gpu_memory_utilization=a.gpu_mem, trust_remote_code=False, seed=0)
sp = SamplingParams(n=a.n, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p, stop_token_ids=EOS_IDS)
outs = llm.generate(prompts, sp)
lens, clipped, ends = [], 0, {1: 0, 250019: 0, "other": 0}
rows = []
for p, o in zip(texts, outs):
    for c in o.outputs:
        L = len(c.token_ids); lens.append(L)
        last = c.token_ids[-1] if c.token_ids else None
        is_clip = (c.finish_reason == "length")
        clipped += is_clip
        ends[last if (last in ends and not is_clip) else "other"] += 1
        rows.append({"prompt": p[:200], "len": L, "finish_reason": c.finish_reason, "last_token": last, "text_tail": c.text[-200:]})
N = len(lens)
summary = {"model": a.model, "n_prompts": len(texts), "n_samples": N, "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p,
           "thinking": a.thinking, "mean_len": statistics.mean(lens), "median_len": statistics.median(lens),
           "p95_len": sorted(lens)[int(0.95 * (N - 1))], "clip_rate": clipped / N,
           "end_EOD_frac": ends[1] / N, "end_EOT_frac": ends[250019] / N, "end_other_frac": ends["other"] / N}
json.dump({"summary": summary, "samples": rows}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
