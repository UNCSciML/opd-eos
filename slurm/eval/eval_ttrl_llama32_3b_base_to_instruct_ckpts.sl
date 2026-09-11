#!/usr/bin/env bash
#SBATCH --job-name=opd_l32_3b_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
# DISK REQUIREMENTS - READ FIRST
# ------------------------------
# Evaluating a 200-step run merges all 10 FSDP checkpoints to HF format first,
# about 60 GB under checkpoint_merged/<run>/ for Llama-3.2-3B. The generations
# themselves are under 1 GB. Export MERGED_ROOT to put the merged models on
# another filesystem, and OUTPUT_ROOT to relocate the generations.
#
# Checkpoint eval for any Llama-3.2-3B -> Llama-3.2-3B-Instruct TTRL run, i.e.
# either of:
#   slurm/train/train_ttrl_llama32_3b_base_to_instruct_baseline_200step.sl
#   slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
#
# This wrapper adds no eval semantics of its own. It only supplies the Slurm
# resources a 3B student needs and then runs the same checkpoint evaluator used
# by every other run in this repository, which reads the student model, rollout stop
# IDs, blocked IDs, thinking flag and prompt template from the train run's
# checkpoint/<run>/run_semantics.json.
#
# Run from the repository root:
#   export CONDA_ENV=/path/to/opd-environment
#   hf auth login          # both Meta checkpoints are gated
#   wandb login
#   export CHECKPOINT_ROOT="$PWD/checkpoint/opd_ttrl_llama32_3b_base_to_instruct_baseline_<train-job-id>"   # or the _semantic_class_ run
#   DRY_RUN=true bash slurm/eval/eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl
#   sbatch --export=ALL slurm/eval/eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl
#
# Adapting this script to another cluster
# ---------------------------------------
# The #SBATCH header above targets four RTX Pro 6000 GPUs on the original
# cluster. Nothing else in this file is site-specific.
#
# 1. Slurm resources. Override them at submit time rather than editing here:
#      CHECKPOINT_ROOT=... sbatch -p YOUR_PARTITION -A YOUR_ACCOUNT --qos YOUR_QOS \
#        --gres=gpu:4 --cpus-per-task=16 --mem=160G --time=3-00:00:00 \
#        --export=ALL slurm/eval/eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl
#    Exporting SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_QOS / SBATCH_GRES /
#    SBATCH_CPUS_PER_TASK / SBATCH_MEM / SBATCH_TIMELIMIT before sbatch works
#    too. Keep four GPUs: the train checkpoints are model_world_size_4_rank_*.pt
#    and generation shards prompts across four independent TP=1 vLLM engines.
# 2. Python environment. export CONDA_ENV=/path/to/opd-environment (a conda env
#    or any prefix containing bin/python). Without it the job uses whatever
#    environment sbatch inherited. If nvcc is not on PATH, export
#    CUDA_MODULE=<your cuda module name>.
# 3. Checkout location. Submit from the repository root, or export
#    PROJECT_ROOT=/path/to/opd-length-inflation and include it in --export.
# 4. Scratch space. FSDP checkpoints are merged to HF format under
#    checkpoint_merged/<run>/ before generation; export MERGED_ROOT to put them
#    on a larger filesystem, and OUTPUT_ROOT to relocate the generations.
# 5. W&B. Run `wandb login`, or export WANDB_MODE=offline and sync afterwards.
#
# Do NOT set EOS_MODE, STOP_TOKEN_IDS or the prompt template by hand here: they
# come from the train run's run_semantics.json so that a run trained under one
# EOS policy can never be silently evaluated under another. That is also why one
# eval file covers every EOS mode of this model pair — point CHECKPOINT_ROOT at
# the baseline run or the semantic_class run and it adapts on its own.
#
# For a deliberate cross-template ablation only, export EVAL_TEMPLATE=dapo (or
# ttrl/eopd) together with a distinct OUTPUT_ROOT.
#
# Formal defaults inherited from the shared evaluator: steps 20,40,...,200;
# AIME24, AIME25 and AMC23 at @16; 8192 generated tokens; four independent TP=1
# vLLM engines with batched prompt shards; one *.tokens.npz beside every JSONL.
# Merged HF checkpoints are written to checkpoint_merged/<run>/; ten merged 3B
# models need roughly 60 GB of scratch space. Set MERGED_ROOT to relocate them.

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi
export PROJECT_ROOT

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${1:-}}
if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "CHECKPOINT_ROOT is required: it is the train run's checkpoint root, e.g." >&2
    echo "  CHECKPOINT_ROOT=\"\$PWD/checkpoint/opd_ttrl_llama32_3b_base_to_instruct_baseline_<train-job-id>\" \\" >&2
    echo "    sbatch --export=ALL slurm/eval/eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl" >&2
    exit 2
fi
export CHECKPOINT_ROOT

exec bash "$PROJECT_ROOT/eval_opd_ttrl_qwen3_1p7b_ckpts20_200_batched_n16_4gpu.sl"
