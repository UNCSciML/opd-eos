#!/usr/bin/env bash
#SBATCH --job-name=opd_ttrl_eos_b
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:4
set -euo pipefail

export TRAIN_TEMPLATE=ttrl
export EOS_MODE=two_stop
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export TOTAL_TRAINING_STEPS=200
export SAVE_FREQ=20
export TEST_FREQ=-1
export ENABLE_THINKING=False
export IS_PLOT=False
export WANDB_MODE=${WANDB_MODE:-online}

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi
exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
