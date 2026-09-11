#!/usr/bin/env python
"""Dump logits / hidden states / greedy tokens for the K2 equivalence test.

--impl official : HF remote code (run inside the k2-reference venv, transformers 5.x)
--impl patched  : HF LlamaForCausalLM on the Llama-view + grouped-RMSNorm patch (training env)
--impl vllm     : vLLM (training env, plugin K2HorizonForCausalLM), last-position full logprobs
All implementations read the same frozen prompt token IDs and write one .npz per run.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

EOS_IDS = [1, 250019]


DEVICE = os.environ.get("K2_DEVICE", "cuda")


def load_hf(impl, model_dir, dtype, attn):
    import transformers
    from transformers import AutoModelForCausalLM
    v5 = int(transformers.__version__.split(".")[0]) >= 5
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    kw = dict(attn_implementation=attn, trust_remote_code=(impl == "official"))
    kw["dtype" if v5 else "torch_dtype"] = dt
    if impl == "patched":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import opd_k2_patch; opd_k2_patch.apply()
    m = AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(DEVICE).eval()
    if impl == "patched":
        n = [mm for mm in m.modules() if type(mm).__name__ == "LlamaRMSNorm"]
        assert n and all(getattr(x, "n_groups", 1) == getattr(m.config, "layernorm_num_groups", 1) for x in n), "grouped-norm patch not active"
    return m


def sampled_positions(L, every, last_k):
    pos = sorted(set(list(range(0, L, every)) + list(range(max(L - last_k, 0), L))))
    return np.asarray(pos)


def run_hf(a):
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    m = load_hf(a.impl, a.model, a.dtype, a.attn)
    prompts = json.load(open(a.prompts))
    layers = [int(x) for x in a.hidden_layers.split(",")]
    caps = {}
    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            caps[name] = o.detach()
        return f
    hs = [m.model.embed_tokens.register_forward_hook(hook("emb")), m.model.norm.register_forward_hook(hook("final_norm"))]
    for li in layers:
        hs.append(m.model.layers[li].register_forward_hook(hook(f"layer{li}")))
    out = {}
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p["token_ids"]], device=DEVICE); L = ids.shape[1]
            am = torch.ones_like(ids); pos_ids = torch.arange(L, device=DEVICE)[None]
            caps.clear()
            t0 = time.time()
            logits = m(input_ids=ids, attention_mask=am, position_ids=pos_ids, use_cache=False).logits[0].float()
            pos = sampled_positions(L, a.positions_every, a.last_k)
            top = torch.topk(logits, 20, dim=-1)
            lsm = torch.log_softmax(logits, dim=-1)
            n = p["name"]
            out[f"{n}/pos"] = pos
            out[f"{n}/logits_pos"] = logits[pos].cpu().numpy().astype(np.float32)
            out[f"{n}/top20_ids"] = top.indices.cpu().numpy().astype(np.int32)
            out[f"{n}/top20_vals"] = top.values.cpu().numpy().astype(np.float32)
            out[f"{n}/eos_logits"] = logits[:, EOS_IDS].cpu().numpy().astype(np.float32)
            out[f"{n}/eos_logp"] = lsm[:, EOS_IDS].cpu().numpy().astype(np.float32)
            out[f"{n}/last_logits"] = logits[-1].cpu().numpy().astype(np.float32)
            for k, v in caps.items():
                out[f"{n}/hid/{k}"] = v[0][pos].float().cpu().numpy().astype(np.float32)
            out[f"{n}/time"] = np.float32(time.time() - t0)
        if a.greedy_n > 0:
            for p in [q for q in prompts if q["kind"] == "ordinary"][: a.greedy_n]:
                ids = torch.tensor([p["token_ids"]], device=DEVICE)
                g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=a.greedy_tokens, eos_token_id=EOS_IDS, pad_token_id=1)
                out[f"{p['name']}/greedy"] = g[0, ids.shape[1]:].cpu().numpy().astype(np.int32)
    for h in hs: h.remove()
    import transformers
    out["meta"] = np.asarray(json.dumps(dict(impl=a.impl, model=a.model, dtype=a.dtype, attn=a.attn, transformers=transformers.__version__, torch=torch.__version__, gpu=(torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"))))
    np.savez(a.out, **out); print("wrote", a.out, "prompts", len(prompts))


def run_vllm(a):
    from vllm import LLM, SamplingParams
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm import ModelRegistry
    assert "K2HorizonForCausalLM" in ModelRegistry.get_supported_archs(), "K2 vLLM plugin not registered"
    prompts = json.load(open(a.prompts)); V = 250624
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_logprobs=V, trust_remote_code=False, seed=0)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=V)
    res = llm.generate([{"prompt_token_ids": p["token_ids"]} for p in prompts], sp)
    out = {}
    for p, r in zip(prompts, res):
        lp = r.outputs[0].logprobs[0]; vec = np.full(V, -np.inf, dtype=np.float32)
        for tid, o in lp.items(): vec[tid] = o.logprob
        out[f"{p['name']}/last_logprobs"] = vec
        out[f"{p['name']}/greedy1"] = np.int32(r.outputs[0].token_ids[0])
    if a.greedy_n > 0:
        sp2 = SamplingParams(max_tokens=a.greedy_tokens, temperature=0.0, stop_token_ids=EOS_IDS)
        ords = [q for q in prompts if q["kind"] == "ordinary"][: a.greedy_n]
        for p, r in zip(ords, llm.generate([{"prompt_token_ids": p["token_ids"]} for p in ords], sp2)):
            out[f"{p['name']}/greedy"] = np.asarray(r.outputs[0].token_ids, dtype=np.int32)
    import vllm
    out["meta"] = np.asarray(json.dumps(dict(impl="vllm", model=a.model, dtype="bf16", vllm=vllm.__version__, torch=torch.__version__, gpu=(torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"))))
    np.savez(a.out, **out); print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["official", "patched", "vllm"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp32"); ap.add_argument("--attn", default="eager")
    ap.add_argument("--hidden-layers", default="0,1,8,18,35"); ap.add_argument("--positions-every", type=int, default=64); ap.add_argument("--last-k", type=int, default=8)
    ap.add_argument("--greedy-n", type=int, default=8); ap.add_argument("--greedy-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=8192); ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    (run_vllm if a.impl == "vllm" else run_hf)(a)


if __name__ == "__main__":
    main()
