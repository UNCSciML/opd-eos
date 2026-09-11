#!/usr/bin/env bash
#SBATCH --job-name=opd_k2_7b_sft1final_sem200
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
# K2-Horizon-7B student = `sft_1_final` <- teacher = main, semantic-EOS-class OPD (EOS_MODE=semantic_class).
# Paired with train_ttrl_k2_7b_sft1final_to_main_baseline_200step.sl: the same stage-control student
# with the EOS fix enabled, so the fix can be read against its own baseline at this stage.
# Everything except the student checkpoint, the EOS mode and the Ray worker-port range is
# inherited from the pretrain_final wrapper.
set -euo pipefail
# Directory holding the derived K2 "Llama-view" model dirs (see docs/K2_HORIZON.md).
K2_MODELS_ROOT=${K2_MODELS_ROOT:-}
if [ -z "$K2_MODELS_ROOT" ]; then
    echo "K2_MODELS_ROOT is required: the directory containing K2-Horizon-7B-<revision>-llamaview" >&2
    echo "See docs/K2_HORIZON.md for the one-time snapshot and Llama-view build." >&2
    exit 2
fi
export K2_MODELS_ROOT
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$K2_MODELS_ROOT/K2-Horizon-7B-sft_1_final-llamaview}
export EOS_MODE=semantic_class
export RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-50000} RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-54999}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
RUN_SUFFIX=${SLURM_JOB_ID:-manual}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_ttrl_k2_7b_sft1final_to_main_semantic_class_${RUN_SUFFIX}}
if [ -n "${PROJECT_ROOT:-}" ]; then PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd); elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd); else PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd); fi
exec bash "$PROJECT_ROOT/slurm/train/train_ttrl_k2_7b_pretrainfinal_to_main_baseline_400step.sl"
