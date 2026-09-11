#!/usr/bin/env bash
#SBATCH --job-name=opd_base_eval_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
export MODEL_LABEL=${MODEL_LABEL:-qwen3_1p7b_base}
export EOS_MODE=${EOS_MODE:-baseline}
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export EVAL_TEMPLATE=${EVAL_TEMPLATE:-ttrl}
export EVAL_TASKS=${EVAL_TASKS:-AMC23}
export EVAL_N=${EVAL_N:-1}
export EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-256}
export EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-1280}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_GROUP=${WANDB_GROUP:-opd_base_eval_smoke}
export RUN_NAME=${RUN_NAME:-smoke_eval_qwen3_1p7b_base_${SLURM_JOB_ID:-local}}

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi
exec bash "$PROJECT_ROOT/slurm/eval/eval_model_4x6000pro.sl"
