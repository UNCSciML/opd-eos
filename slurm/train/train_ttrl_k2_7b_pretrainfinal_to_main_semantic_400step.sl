#!/usr/bin/env bash
#SBATCH --job-name=opd_k2_7b_sem400
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
# K2-Horizon-7B pretraining-final student <- K2-Horizon-7B final post-trained teacher,
# TTRL, semantic-EOS-class OPD (EOS_MODE=semantic_class: same stop set, termination supervised on p(1)+p(250019)).
#
# Model pair (revision-pinned; regenerate the metadata table with scripts/diagnostics/check_k2_eos.py):
#   student  IFM/K2-Horizon-7B @ pretrain_final  sha fdbc38e9037842fe12f3b9967647dbd6b9714dd8
#   teacher  IFM/K2-Horizon-7B @ main            sha 586b03f0fd1fbbf2f13eeafc33749e95ae34dd10
# Both are loaded through derived "Llama-view" directories (scripts/diagnostics/make_k2_llama_view.py):
# untouched safetensors + Llama config carrying layernorm_num_groups=4, the *main* tokenizer view
# (vocab/IDs/merges identical across revisions; 24 special-token surface names renamed) and a derived
# chat template with an enable_thinking=False switch (empty <ifm|think>\n</ifm|think> block).
# The grouped RMSNorm is restored by scripts/k2/opd_k2_patch.py (HF, env-gated) and the vLLM plugin
# scripts/k2/vllm_plugin (pip install -e).  Run scripts/k2/install_k2_hooks.sh once per environment.
#
# Differences from the Llama/Gemma wrappers (all deliberate, see docs/K2_HORIZON.md):
#   * context stays within the student's native 8192 (prompt 1024 + response 7168); rope_theta untouched
#   * REWARD_MICRO_BATCH_SIZE=4 and --mem=400G because student and teacher are ~9B with a 250k vocab
#   * OPD_K2_GROUPED_RMSNORM=1 activates the grouped-norm patch in every HF worker
set -euo pipefail

# Directory holding the derived K2 "Llama-view" model dirs (see docs/K2_HORIZON.md).
K2_MODELS_ROOT=${K2_MODELS_ROOT:-}
if [ -z "$K2_MODELS_ROOT" ]; then
    echo "K2_MODELS_ROOT is required: the directory containing K2-Horizon-7B-<revision>-llamaview" >&2
    echo "See docs/K2_HORIZON.md for the one-time snapshot and Llama-view build." >&2
    exit 2
fi
export K2_MODELS_ROOT
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$K2_MODELS_ROOT/K2-Horizon-7B-pretrain_final-llamaview}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-$K2_MODELS_ROOT/K2-Horizon-7B-main-llamaview}
export CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$REWARD_MODEL_PATH}
export TRAIN_TEMPLATE=ttrl
export EOS_MODE=semantic_class
export ENABLE_THINKING=False
export OPD_K2_GROUPED_RMSNORM=1

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export ROLLOUT_N=${ROLLOUT_N:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export REWARD_MAX_TOKEN_LEN_PER_GPU=${REWARD_MAX_TOKEN_LEN_PER_GPU:-8192}
export REWARD_MICRO_BATCH_SIZE=${REWARD_MICRO_BATCH_SIZE:-4}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:--1}
export IS_PLOT=False
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}

export RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-30000}
export RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-34999}

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi
export PYTHONPATH="$PROJECT_ROOT/scripts/k2:${PYTHONPATH:-}"

RUN_SUFFIX=${SLURM_JOB_ID:-manual}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_ttrl_k2_7b_pretrainfinal_to_main_semantic_class_${RUN_SUFFIX}}
export LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/$EXPERIMENT_NAME}
export EOS_DIAGNOSTIC_OUTPUT_DIR=${EOS_DIAGNOSTIC_OUTPUT_DIR:-$PROJECT_ROOT/diagnostics/$EXPERIMENT_NAME}

exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
