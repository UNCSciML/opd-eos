#!/usr/bin/env bash
#SBATCH --job-name=opd_eos_tpl_smoke
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --array=0-6%1

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
CASE_ID=${SLURM_ARRAY_TASK_ID:?This smoke script must run as a Slurm array}

case "$CASE_ID" in
    0) TRAIN_TEMPLATE=ttrl; EOS_MODE=baseline ;;
    1) TRAIN_TEMPLATE=ttrl; EOS_MODE=two_stop ;;
    2) TRAIN_TEMPLATE=ttrl; EOS_MODE=teacher_map ;;
    3) TRAIN_TEMPLATE=ttrl; EOS_MODE=semantic_class ;;
    4) TRAIN_TEMPLATE=ttrl; EOS_MODE=canonical ;;
    5) TRAIN_TEMPLATE=dapo; EOS_MODE=baseline ;;
    6) TRAIN_TEMPLATE=eopd; EOS_MODE=baseline ;;
    *) echo "Unsupported smoke case: $CASE_ID" >&2; exit 2 ;;
esac

export TRAIN_TEMPLATE EOS_MODE
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_N=1
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=256
export MAX_MODEL_LEN=1280
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export TOTAL_TRAINING_STEPS=1
export SAVE_FREQ=-1
export TEST_FREQ=-1
export REWARD_MICRO_BATCH_SIZE=1
export IS_PLOT=False
export ENABLE_THINKING=False
export WANDB_MODE=${WANDB_MODE:-online}
export EXPERIMENT_NAME="smoke_opd_${TRAIN_TEMPLATE}_${EOS_MODE}_q17b_q4b_4x6000pro_${SLURM_ARRAY_JOB_ID}_${CASE_ID}"
export LOG_DIR="$PROJECT_ROOT/logs/opd_eos_template_smoke/runtime_${SLURM_ARRAY_JOB_ID}_${CASE_ID}"

echo "smoke_case=$CASE_ID train_template=$TRAIN_TEMPLATE eos_mode=$EOS_MODE"
bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
