#!/usr/bin/env python
"""Re-package a verl-merged K2 'Llama-view' checkpoint into the official K2-Horizon format.

The Llama-view dir (config: model_type=llama + layernorm_num_groups) only works with our grouped-RMSNorm
patch; stock transformers would silently use ungrouped RMSNorm.  Weight names are identical to the
official checkpoint (the view symlinks the official shards), so the export is: official config + remote
code + generation_config, tokenizer (main's), the training chat template, and the merged shards.
"""
import argparse, json, os, shutil
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--merged", required=True); ap.add_argument("--official", required=True)
ap.add_argument("--tokenizer-ref", required=True, help="official snapshot whose tokenizer files must match (main)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def shapes(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]; out = {}
    for shard in sorted(set(idx.values())):
        with safe_open(os.path.join(d, shard), "pt") as f:
            for k in f.keys(): out[k] = tuple(f.get_slice(k).get_shape())
    return out
m, o = shapes(a.merged), shapes(a.official)
assert set(m) == set(o), f"weight-name mismatch: only-merged={sorted(set(m)-set(o))[:5]} only-official={sorted(set(o)-set(m))[:5]}"
bad = [k for k in m if m[k] != o[k]]; assert not bad, f"shape mismatch: {bad[:5]}"
print(f"weights: {len(m)} tensors, names+shapes identical to official")

# vocabulary must be byte-identical to the reference (main) snapshot; tokenizer_config.json is taken from the
# reference itself (official transformers-5 format: TokenizersBackend) instead of the transformers-4.56 file that
# verl wrote next to the merged weights (same content semantically; differs only in format keys).
src, ref = os.path.join(a.merged, "tokenizer.json"), os.path.join(a.tokenizer_ref, "tokenizer.json")
assert open(src, "rb").read() == open(ref, "rb").read(), "tokenizer.json differs from reference snapshot"
shutil.copy2(ref, a.out); shutil.copy2(os.path.join(a.tokenizer_ref, "tokenizer_config.json"), a.out)
print("tokenizer.json identical to reference; tokenizer_config.json taken from", a.tokenizer_ref)

cfg = json.load(open(os.path.join(a.official, "config.json")))
cfg.pop("torch_dtype", None); cfg["dtype"] = "bfloat16"
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
for f in ["configuration_k2_horizon.py", "modeling_k2_horizon.py", "generation_config.json"]:
    shutil.copy2(os.path.join(a.official, f), a.out)
shutil.copy2(os.path.join(a.merged, "chat_template.jinja"), os.path.join(a.out, "chat_template.jinja"))
off_tpl = os.path.join(a.tokenizer_ref, "chat_template.jinja")
if os.path.exists(off_tpl): shutil.copy2(off_tpl, os.path.join(a.out, "chat_template.official.jinja"))
idx = json.load(open(os.path.join(a.merged, "model.safetensors.index.json")))
for shard in sorted(set(idx["weight_map"].values())):
    dst = os.path.join(a.out, shard)
    if os.path.exists(dst): os.remove(dst)
    try: os.link(os.path.join(a.merged, shard), dst)
    except OSError: shutil.copy2(os.path.join(a.merged, shard), dst)
shutil.copy2(os.path.join(a.merged, "model.safetensors.index.json"), a.out)
print("exported ->", a.out, sorted(os.listdir(a.out)))
