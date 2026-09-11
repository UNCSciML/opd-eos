#!/usr/bin/env python
"""Compare two k2_equivalence.py dumps (HF vs HF, or HF vs vLLM) and print a markdown report."""
import argparse, json
import numpy as np

EOS = [1, 250019]


def names(d):
    return sorted({k.split("/")[0] for k in d.files if "/" in k and not k.startswith("meta")})


def softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()


def cmp_hf(A, B, label):
    rows = []; agg = dict(maxerr=[], meanerr=[], cos=[], top1=[], top5=[], top20=[], greedy=[])
    hid = {}
    for n in names(A):
        if f"{n}/logits_pos" not in A or f"{n}/logits_pos" not in B: continue
        la, lb = A[f"{n}/logits_pos"], B[f"{n}/logits_pos"]
        err = np.abs(la - lb); cos = float(np.mean([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) for x, y in zip(la, lb)]))
        ta, tb = A[f"{n}/top20_ids"], B[f"{n}/top20_ids"]
        top1 = float(np.mean(ta[:, 0] == tb[:, 0])); top5 = float(np.mean([len(set(x[:5]) & set(y[:5])) / 5 for x, y in zip(ta, tb)])); top20 = float(np.mean([len(set(x) & set(y)) / 20 for x, y in zip(ta, tb)]))
        pa, pb = softmax(A[f"{n}/last_logits"]), softmax(B[f"{n}/last_logits"])
        eos = " ".join(f"logit({i}) {A[f'{n}/last_logits'][i]:.3f}/{B[f'{n}/last_logits'][i]:.3f} p({i}) {pa[i]:.3e}/{pb[i]:.3e}" for i in EOS) + f" pSTOP {pa[EOS].sum():.3e}/{pb[EOS].sum():.3e}"
        g = ""
        if f"{n}/greedy" in A and f"{n}/greedy" in B:
            ga, gb = A[f"{n}/greedy"], B[f"{n}/greedy"]; L = min(len(ga), len(gb)); same = ga[:L] == gb[:L]
            first = int(np.argmin(same)) if not same.all() else L; g = f"{first}/{L} identical-prefix ({'ALL' if same.all() and len(ga)==len(gb) else 'diverge@'+str(first)})"; agg["greedy"].append(first / max(L, 1))
        for k in [f for f in A.files if f.startswith(f"{n}/hid/")]:
            if k in B: hid.setdefault(k.split("/")[-1], []).append(float(np.abs(A[k] - B[k]).max()))
        rows.append((n, la.shape[0], float(err.max()), float(err.mean()), cos, top1, top5, top20, g, eos))
        for k, v in zip(["maxerr", "meanerr", "cos", "top1", "top5", "top20"], [err.max(), err.mean(), cos, top1, top5, top20]): agg[k].append(float(v))
    print(f"\n### {label}\n")
    print("| prompt | #pos | max|Δlogit| | mean|Δlogit| | cos | top-1 | top-5 | top-20 | greedy | last-position EOS (A/B) |"); print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows: print(f"| {r[0]} | {r[1]} | {r[2]:.3e} | {r[3]:.3e} | {r[4]:.6f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]} | {r[9]} |")
    print(f"\n**Overall:** max|Δlogit| = {max(agg['maxerr']):.3e}, mean|Δlogit| = {np.mean(agg['meanerr']):.3e}, cos = {np.mean(agg['cos']):.6f}, top-1 = {np.mean(agg['top1']):.4f}, top-5 = {np.mean(agg['top5']):.4f}, top-20 = {np.mean(agg['top20']):.4f}" + (f", greedy identical-prefix fraction = {np.mean(agg['greedy']):.3f}" if agg["greedy"] else ""))
    if hid:
        print("\nhidden-state max|Δ| by layer (mean over prompts): " + ", ".join(f"{k}: {np.mean(v):.3e}" for k, v in sorted(hid.items(), key=lambda kv: (kv[0] != 'emb', kv[0] == 'final_norm', kv[0]))))
    return agg


def cmp_vllm(A, Vd, label):
    print(f"\n### {label}\n"); print("| prompt | max|Δlogp| (full vocab) | mean|Δlogp| | top-1 same | top-20 overlap | p(1) HF/vLLM | p(250019) HF/vLLM | p(STOP) HF/vLLM |"); print("|---|---|---|---|---|---|---|---|")
    for n in names(A):
        if f"{n}/last_logprobs" not in Vd: continue
        la = A[f"{n}/last_logits"]; lpa = la - la.max() - np.log(np.exp(la - la.max()).sum()); lpv = Vd[f"{n}/last_logprobs"]
        ok = np.isfinite(lpv); d = np.abs(lpa[ok] - lpv[ok]); pa, pv = np.exp(lpa), np.exp(np.where(ok, lpv, -np.inf))
        ta, tv = np.argsort(-lpa)[:20], np.argsort(-np.where(ok, lpv, -np.inf))[:20]
        print(f"| {n} | {d.max():.3e} | {d.mean():.3e} | {ta[0]==tv[0]} | {len(set(ta)&set(tv))/20:.2f} | {pa[1]:.3e}/{pv[1]:.3e} | {pa[250019]:.3e}/{pv[250019]:.3e} | {pa[EOS].sum():.3e}/{pv[EOS].sum():.3e} |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--label", default="")
    x = ap.parse_args(); A, B = np.load(x.a), np.load(x.b)
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    print(f"A = {ma}\nB = {mb}")
    (cmp_vllm if mb.get("impl") == "vllm" else cmp_hf)(A, B, x.label or f"{ma['impl']}/{ma.get('dtype')}/{ma.get('attn','')} vs {mb['impl']}/{mb.get('dtype')}/{mb.get('attn','')}")


if __name__ == "__main__":
    main()
