#!/usr/bin/env python
"""Phase-1 EOS probe: next-token termination mass at frozen terminal prefixes (official K2 HF impl).

For every prefix: p_eod=p(1), p_eot=p(250019), p_stop=p_eod+p_eot, eot_share=p_eot/p_stop,
raw logits for ids 1/250019, top-1 token.  Writes per-example CSV + JSON summary per prefix set.
"""
import argparse, json, csv, os, sys, time
from pathlib import Path
import numpy as np
import torch

EOD, EOT = 1, 250019


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--prefixes", action="append", required=True, help="LABEL=prefix_set.json")
    ap.add_argument("--impl", choices=["official", "patched"], default="official")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from k2_equivalence import load_hf
    m = load_hf(a.impl, a.model, a.dtype, a.attn)
    tok_name = Path(a.model).name
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for spec in a.prefixes:
        label, path = spec.split("=", 1); pre = json.load(open(path))["prefixes"]
        rows = []
        with torch.no_grad():
            for p in pre:
                ids = torch.tensor([p["token_ids"]], device="cuda")
                logits = m(input_ids=ids, use_cache=False).logits[0, -1].float()
                lp = torch.log_softmax(logits, -1); pr = lp.exp()
                p_eod, p_eot = float(pr[EOD]), float(pr[EOT]); p_stop = p_eod + p_eot
                top = torch.topk(pr, 5)
                rows.append(dict(idx=p["idx"], task=p.get("task"), example_id=p.get("example_id"), n_tokens=p["n_tokens"],
                                 logit_eod=float(logits[EOD]), logit_eot=float(logits[EOT]), logp_eod=float(lp[EOD]), logp_eot=float(lp[EOT]),
                                 p_eod=p_eod, p_eot=p_eot, p_stop=p_stop, eot_share=(p_eot / p_stop if p_stop > 0 else float("nan")),
                                 top1_id=int(top.indices[0]), top1_p=float(top.values[0]), top5_ids=" ".join(str(int(i)) for i in top.indices)))
        with open(out / f"{a.name}__{label}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        def stats(k):
            v = np.array([r[k] for r in rows], dtype=np.float64); v = v[np.isfinite(v)]
            return dict(mean=float(v.mean()), median=float(np.median(v)), std=float(v.std()), p10=float(np.quantile(v, .1)), p25=float(np.quantile(v, .25)), p75=float(np.quantile(v, .75)), p90=float(np.quantile(v, .9)), n=int(len(v)))
        summary[label] = {k: stats(k) for k in ["p_eod", "p_eot", "p_stop", "eot_share", "logit_eod", "logit_eot"]}
        summary[label]["frac_top1_is_eod"] = float(np.mean([r["top1_id"] == EOD for r in rows])); summary[label]["frac_top1_is_eot"] = float(np.mean([r["top1_id"] == EOT for r in rows])); summary[label]["frac_top1_is_stop"] = float(np.mean([r["top1_id"] in (EOD, EOT) for r in rows]))
        s = summary[label]; print(f"[{a.name}] {label}: n={s['p_eod']['n']} p_eod mean {s['p_eod']['mean']:.4f} med {s['p_eod']['median']:.4f} | p_eot mean {s['p_eot']['mean']:.4f} med {s['p_eot']['median']:.4f} | p_stop mean {s['p_stop']['mean']:.4f} | eot_share mean {s['eot_share']['mean']:.3f} med {s['eot_share']['median']:.3f} | top1∈STOP {s['frac_top1_is_stop']:.2f}")
    json.dump(dict(model=a.model, name=a.name, impl=a.impl, dtype=a.dtype, attn=a.attn, summary=summary), open(out / f"{a.name}__summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
