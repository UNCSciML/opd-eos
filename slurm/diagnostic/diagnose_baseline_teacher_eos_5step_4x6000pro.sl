#!/usr/bin/env bash
#SBATCH --job-name=opd_eos_diag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4
set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

run_tag=${SLURM_JOB_ID:-dryrun}
export PROJECT_ROOT
export CONDA_ENV=${CONDA_ENV:-}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Qwen/Qwen3-4B}
export TRAIN_TEMPLATE=ttrl
export EOS_MODE=baseline
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export TOTAL_TRAINING_STEPS=5
export SAVE_FREQ=-1
export TEST_FREQ=-1
export CRITIC_WARMUP=6
export ENABLE_THINKING=False
export IS_PLOT=False
export WANDB_MODE=disabled
export EXPERIMENT_NAME="opd_ttrl_baseline_teacher_eos_diagnostic_${run_tag}"
export EOS_DIAGNOSTIC_OUTPUT_DIR="$PROJECT_ROOT/diagnostics/teacher_eos_at_student_eos_${run_tag}"

exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
