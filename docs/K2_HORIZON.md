# K2-Horizon-7B OPD extension — how to run it (portable)

Student `IFM/K2-Horizon-7B@pretrain_final` (sha `fdbc38e9037842fe12f3b9967647dbd6b9714dd8`, "pretraining-final student")
Teacher `IFM/K2-Horizon-7B@main` (sha `586b03f0fd1fbbf2f13eeafc33749e95ae34dd10`, "final post-trained teacher"; byte-identical to tag `sft_2_final`; `rl_merged` is a separate branch).

## Why a derived model view is needed
* K2's remote code targets transformers ≥ 5.13; the OPD environment is transformers 4.56 / vLLM 0.11 (no `K2HorizonForCausalLM`).
* K2-7B dense is architecturally Llama (0 experts, all-MLP layers, no q/k-norm, no sliding window, silu, head_dim 128, default RoPE)
  **except** its RMSNorm normalises over `layernorm_num_groups = 4` groups of the hidden dimension.
* We therefore load the untouched weights through a **Llama-view** directory (Llama config + `layernorm_num_groups`, symlinked
  safetensors, the `main` tokenizer, a derived chat template with an `enable_thinking=False` switch) and restore the grouped
  norm with two hooks. Verified bit-exact against the official implementation (FP32 and BF16), vLLM within backend noise.

## One-time setup (any cluster)
```bash
# 1) snapshots (revision-pinned, untouched) -> a large filesystem
python - <<'PY'
from huggingface_hub import snapshot_download
for rev in ["pretrain_final", "main"]:
    snapshot_download("IFM/K2-Horizon-7B", revision=rev, local_dir=f"$K2_MODELS_ROOT/K2-Horizon-7B-{rev}")
PY
# 2) derived Llama-views (main tokenizer for both)
for rev in pretrain_final main; do
  python scripts/diagnostics/make_k2_llama_view.py --src $K2_MODELS_ROOT/K2-Horizon-7B-$rev \
      --tokenizer-src $K2_MODELS_ROOT/K2-Horizon-7B-main --out $K2_MODELS_ROOT/K2-Horizon-7B-$rev-llamaview
done
# 3) hooks in the training environment: vLLM plugin + env-gated HF grouped-RMSNorm patch
bash scripts/k2/install_k2_hooks.sh
```
`scripts/diagnostics/check_k2_eos.py` prints the metadata / EOS table; `scripts/k2/k2_equivalence.py` + `k2_equivalence_compare.py`
reproduce the equivalence test (needs a transformers ≥ 5.15 venv for the official side); `scripts/k2/k2_eos_probe.py` +
`k2_probe_report.py` reproduce the stage-wise EOS probe.

## Training / evaluation entrypoints
```bash
export K2_MODELS_ROOT=/path/to/models     # required by the wrappers; ~18 GB per revision plus its view
# vanilla OPD: official rollout stop set {1, 250019}, EOS tokens supervised separately (EOS_MODE=baseline)
sbatch [site resources] slurm/train/train_ttrl_k2_7b_pretrainfinal_to_main_baseline_400step.sl
# semantic-EOS OPD: same stop set, termination supervised on p(1)+p(250019) (EOS_MODE=semantic_class)
sbatch [site resources] slurm/train/train_ttrl_k2_7b_pretrainfinal_to_main_semantic_400step.sl
# 20-checkpoint eval, steps 20..400 (template from run_semantics.json; K2 context limits 7168 / 8192)
# add FINAL_STEP=200 for the mid_1_final / sft_1_final stage controls
CHECKPOINT_ROOT=checkpoint/<run> sbatch --export=ALL [site resources] slurm/eval/eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl
# stage controls: the same student family at two later pre-training stages, 200 updates each
sbatch [site resources] slurm/train/train_ttrl_k2_7b_mid1final_to_main_{baseline,semantic}_200step.sl
sbatch [site resources] slurm/train/train_ttrl_k2_7b_sft1final_to_main_{baseline,semantic}_200step.sl
CHECKPOINT_ROOT=checkpoint/<run> FINAL_STEP=200 sbatch --export=ALL [site resources] \
  slurm/eval/eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl
```
The one eval entrypoint serves every K2 run: the student, template, stop/blocked ids, thinking flag and
EOS mode all come from that run's `run_semantics.json`, so only `CHECKPOINT_ROOT` (and `FINAL_STEP` for the
200-update stage controls) changes.

Both wrappers export `OPD_K2_GROUPED_RMSNORM=1` and `PYTHONPATH=scripts/k2`; everything else (dataset, TTRL text, lr 1e-6,
bs 16, n=4, T=1.0, save every 20, sampled-token OPD) is identical to the Llama/Gemma wrappers.

**Training horizon.** The `pretrain_final -> main` pair is the long-horizon arm of the study and trains for
**400 updates** (checkpoints `20,40,...,400`), twice the 200-update horizon used for Qwen, Llama and Gemma.
Its two wrappers are named `..._400step.sl` accordingly, and the checkpoint evaluator defaults to the matching
20-point curve. The stage-control students (`mid_1_final`, `sft_1_final`) keep the 200-update horizon so they
stay directly comparable with the other model families; their wrappers are named `..._200step.sl` and their
eval needs `FINAL_STEP=200`. Both horizons are just `TOTAL_TRAINING_STEPS` / `FINAL_STEP` overrides.

## Deliberate differences from Llama/Gemma
| item | K2 | Llama/Gemma |
|---|---|---|
| context | prompt 1024 + response 7168 = 8192 (student native `max_position_embeddings`); eval 7168 / 8192 | same train; eval 8192 / 12288 |
| model size | ≈ 9.0 B real params each (250 624 vocab) → `REWARD_MICRO_BATCH_SIZE=4`, host RAM 512 G, prefer H200/B200/RTX-Pro-6000 | 3–4 B |
| EOS | `generation_config.eos_token_id=[1,250019]` for every revision → `EOS_MODE=baseline` already decodes with the official two-token stop set | student native set = single EOS |
| template | K2 chat template (thinking by default) + derived `enable_thinking=False` → `<ifm|think>\n</ifm|think>` | Instruct/IT templates |

## Key facts from the stage probe
* Tokenizer: vocab / merges / IDs identical across revisions; 24 special-token surface strings renamed → use `main`'s tokenizer.
* Teacher sanity (`scripts/k2/k2_teacher_sanity.py`; non-thinking, greedy ≤ 4096): 70 % terminate (always EOT), AMC23 87.5 %, AIME24 43.3 %, p(EOT) at end 0.98.
* Stage-wise probe (`scripts/k2/k2_eos_probe.py` + `k2_probe_report.py`): `pretrain_final` p(EOD) 0.94 / EOT-share 0.00 → flip complete at `mid_1_final` (EOT-share 0.995) → `main`
  p(EOD) 0.000, EOT-share 1.00. Case A (strong natural mismatch).
