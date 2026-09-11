#!/usr/bin/env bash
#SBATCH --job-name=opd_base_s0_tpl
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-1%2

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

CASE_ID=${SLURM_ARRAY_TASK_ID:?This entrypoint must run as a Slurm array}
case "$CASE_ID" in
    0) EVAL_TEMPLATE=ttrl ;;
    1) EVAL_TEMPLATE=dapo ;;
    *) echo "Unsupported template case: $CASE_ID" >&2; exit 2 ;;
esac

export PROJECT_ROOT
export CONDA_ENV=${CONDA_ENV:-}
export MODEL_PATH=Qwen/Qwen3-1.7B-Base
export MODEL_LABEL=qwen3_1p7b_base_step0
export EOS_MODE=baseline
export STOP_TOKEN_IDS=
export BLOCKED_TOKEN_IDS=
export ENABLE_THINKING=false
export EVAL_TEMPLATE
export EVAL_TASKS=AIME24,AIME25,AMC23
export EVAL_N=16
export EVAL_MAX_TOKENS=8192
export EVAL_MAX_MODEL_LEN=12288
export EVAL_TEMPERATURE=0.7
export EVAL_TOP_P=0.95
export EVAL_SEED=0
export GPU_MEMORY_UTILIZATION=0.85
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}
export WANDB_GROUP=opd_length_inflation_base_step0_template_eval
export RUN_NAME="opd_length_inflation_qwen3_1p7b_base_step0_${EVAL_TEMPLATE}_n16_${SLURM_JOB_ID:-local}"
export OUTPUT_ROOT="$PROJECT_ROOT/eval_outputs/$RUN_NAME"

exec bash "$PROJECT_ROOT/slurm/eval/eval_model_4x6000pro.sl"
