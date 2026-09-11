#!/usr/bin/env bash
#SBATCH --job-name=opd_g3_4b_base200
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=296G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:4
# DISK REQUIREMENTS - READ FIRST
# ------------------------------
# A 200-step run with SAVE_FREQ=20 keeps 10 full training checkpoints (weights
# plus optimizer state). Measured on the completed Gemma-3-4B run: 48 GB per
# checkpoint, 475 GB per run for Gemma-3-4B. Evaluating that run later merges
# every checkpoint to HF format, adding about 80 GB under checkpoint_merged/.
# Confirm the space before submitting. Set SAVE_FREQ=200 to keep only the final
# checkpoint, which costs you the 10-point eval curve.
#
# Gemma-3-4B-PT -> Gemma-3-4B-IT TTRL baseline (no EOS fix).
#
# Run from the repository root:
#   export CONDA_ENV=/path/to/opd-environment
#   hf auth login          # accept the Gemma model licenses first
#   wandb login
#   DRY_RUN=true bash slurm/train/train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl
#   sbatch slurm/train/train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl
#
# Adapting this script to another cluster
# ---------------------------------------
# The #SBATCH header above targets four RTX Pro 6000 GPUs on the original
# cluster. Nothing else in this file is site-specific.
#
# 1. Slurm resources. Override them at submit time rather than editing here:
#      sbatch -p YOUR_PARTITION -A YOUR_ACCOUNT --qos YOUR_QOS \
#        --gres=gpu:4 --cpus-per-task=16 --mem=296G --time=1-00:00:00 \
#        --export=ALL slurm/train/train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl
#    Exporting SBATCH_PARTITION / SBATCH_ACCOUNT / SBATCH_QOS / SBATCH_GRES /
#    SBATCH_CPUS_PER_TASK / SBATCH_MEM / SBATCH_TIMELIMIT before sbatch works
#    too. Keep four GPUs: checkpoints are saved as model_world_size_4_rank_*.pt
#    and the evaluator expects that layout.
# 2. Python environment. export CONDA_ENV=/path/to/opd-environment (a conda env
#    or any prefix containing bin/python). Without it the job uses whatever
#    environment sbatch inherited. If nvcc is not on PATH, export
#    CUDA_MODULE=<your cuda module name>.
# 3. Checkout location. Submit from the repository root, or export
#    PROJECT_ROOT=/path/to/opd-length-inflation and include it in --export.
# 4. Models. The defaults are Hugging Face IDs. On a compute node without
#    Internet access, point ACTOR_MODEL_PATH and REWARD_MODEL_PATH at local
#    model directories instead.
# 5. W&B. Run `wandb login`, or export WANDB_MODE=offline and sync afterwards.
#
# This script never edits config.json, generation_config.json or tokenizer
# files. EOS behaviour is process-local and is recorded in run_semantics.json.

set -euo pipefail

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-google/gemma-3-4b-pt}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-google/gemma-3-4b-it}
export CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$REWARD_MODEL_PATH}
export TRAIN_TEMPLATE=ttrl
export EOS_MODE=baseline
export ENABLE_THINKING=False
# Gemma-3's processor emits variable-length token_type_ids even for text-only
# samples. They are optional for text-only forward passes and must not be
# treated as image/video tensors after rollout responses are appended.
export RETURN_MULTI_MODAL_INPUTS=False
# The async activation-offload scheduler counts the unused vision FSDP layers
# and cannot reload text-only Gemma checkpoints in the matching backward order.
export ENABLE_ACTIVATION_OFFLOAD=False

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
export ROLLOUT_N=${ROLLOUT_N:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export REWARD_MAX_TOKEN_LEN_PER_GPU=${REWARD_MAX_TOKEN_LEN_PER_GPU:-8192}
export REWARD_MICRO_BATCH_SIZE=${REWARD_MICRO_BATCH_SIZE:-8}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:--1}
export IS_PLOT=False
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}

# These disjoint ranges let the Llama and Gemma jobs share a larger node safely.
# On a four-GPU-exclusive node they have no effect on the training semantics.
export RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-25000}
export RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-29999}

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

RUN_SUFFIX=${SLURM_JOB_ID:-manual}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_ttrl_gemma3_4bpt_to_4bit_baseline_${RUN_SUFFIX}}
export LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/$EXPERIMENT_NAME}
export EOS_DIAGNOSTIC_OUTPUT_DIR=${EOS_DIAGNOSTIC_OUTPUT_DIR:-$PROJECT_ROOT/diagnostics/$EXPERIMENT_NAME}

# The shared launcher's filename is historical; every model-specific value above
# is passed explicitly and the launcher does not edit downloaded model configs.
exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
