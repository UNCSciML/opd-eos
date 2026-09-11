#!/usr/bin/env bash
#SBATCH --job-name=opd_k2_7b_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
# K2-Horizon-7B pretraining-final -> final post-trained checkpoint eval (any EOS mode).
# Same shared evaluator as Llama/Gemma; the ONLY differences are the K2 context limits:
#   EVAL_MAX_TOKENS=7168, EVAL_MAX_MODEL_LEN=8192   (student native context = 8192; Llama/Gemma use 8192/12288)
# and the vLLM K2 plugin (scripts/k2/vllm_plugin) which must be installed in the environment.
# Merged HF checkpoints (~18 GB each x 10) are written under MERGED_ROOT.
# NOTE: forced (not ${EVAL_MAX_TOKENS:-...}) because slurm/byu_opdr_eval.sbatch exports the generic
# default EVAL_MAX_TOKENS=8192 before exec'ing this script; with max_model_len=8192 that budget can
# never be reached and the evaluator's truncation rule (len >= max_tokens) silently reports 0.
# Override only via K2_EVAL_MAX_TOKENS / K2_EVAL_MAX_MODEL_LEN.
export EVAL_MAX_TOKENS=${K2_EVAL_MAX_TOKENS:-7168}
export EVAL_MAX_MODEL_LEN=${K2_EVAL_MAX_MODEL_LEN:-8192}
export OPD_K2_GROUPED_RMSNORM=1
set -euo pipefail

# Checkpoint list: steps SAVE_FREQ..FINAL_STEP. Set FINAL_STEP=200 for the mid_1_final /
# sft_1_final stage controls, or to the horizon a short run actually reached; or pass STEPS.
FINAL_STEP=${FINAL_STEP:-400}
SAVE_FREQ=${SAVE_FREQ:-20}
if [ -z "${STEPS:-}" ]; then
    STEPS=$(seq "$SAVE_FREQ" "$SAVE_FREQ" "$FINAL_STEP" | paste -sd,)
fi
export STEPS

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
    echo "  CHECKPOINT_ROOT=\"\$PWD/checkpoint/opd_ttrl_k2_7b_pretrainfinal_to_main_baseline_<train-job-id>\" \\" >&2
    echo "    sbatch --export=ALL slurm/eval/eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl" >&2
    exit 2
fi
export CHECKPOINT_ROOT

exec bash "$PROJECT_ROOT/eval_opd_ttrl_qwen3_1p7b_ckpts20_200_batched_n16_4gpu.sl"
