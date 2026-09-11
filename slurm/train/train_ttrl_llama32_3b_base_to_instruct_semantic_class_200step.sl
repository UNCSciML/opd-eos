#!/usr/bin/env bash
#SBATCH --job-name=opd_l32_3b_sem200
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:4
# DISK REQUIREMENTS - READ FIRST
# ------------------------------
# A 200-step run with SAVE_FREQ=20 keeps 10 full training checkpoints (weights
# plus optimizer state). Measured on the completed Gemma-3-4B run: 48 GB per
# checkpoint, roughly 350 GB per run for Llama-3.2-3B. Evaluating that run later merges
# every checkpoint to HF format, adding about 60 GB under checkpoint_merged/.
# Confirm the space before submitting. Set SAVE_FREQ=200 to keep only the final
# checkpoint, which costs you the 10-point eval curve.
#
# Llama-3.2-3B -> Llama-3.2-3B-Instruct TTRL run with the semantic-EOS-class fix.
#
# Run from the repository root:
#   export CONDA_ENV=/path/to/opd-environment
#   hf auth login          # both Meta checkpoints are gated
#   wandb login
#   DRY_RUN=true bash slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
#   sbatch slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
#
# Why semantic_class applies to this pair
# ---------------------------------------
# Llama-3.2-3B (student) declares a single terminal token, <|end_of_text|>
# (128001). Llama-3.2-3B-Instruct — the teacher and the chat-template source —
# declares three: <|end_of_text|> (128001), <|eom_id|> (128008) and <|eot_id|>
# (128009), and its template ends an assistant turn with <|eot_id|>. So the
# teacher puts termination mass on a token the baseline student never treats as
# terminal, which is the same pathology as the Qwen pair.
#
# The token ids are NOT hard-coded here. run_semantics.py discovers both native
# sets at submit time and stores their ordered union as the semantic terminal
# set; semantic_class then compares the aggregated probability of that whole set
# on the teacher and student side, and all three tokens stop the rollout.
#
# Because the union has three tokens, the pairwise-only modes teacher_map and
# canonical are NOT available for this pair and fail preflight with an explicit
# error. baseline, two_stop and semantic_class are the usable modes.
#
# Adapting this script to another cluster
# ---------------------------------------
# The #SBATCH header above targets four RTX Pro 6000 GPUs on the original
# cluster. Nothing else in this file is site-specific.
#
# 1. Slurm resources. Override them at submit time rather than editing here:
#      sbatch -p YOUR_PARTITION -A YOUR_ACCOUNT --qos YOUR_QOS \
#        --gres=gpu:4 --cpus-per-task=16 --mem=240G --time=1-00:00:00 \
#        --export=ALL slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
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

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-meta-llama/Llama-3.2-3B}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}
export CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$REWARD_MODEL_PATH}
export TRAIN_TEMPLATE=ttrl
export EOS_MODE=semantic_class
export ENABLE_THINKING=False

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
export RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
export RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-24999}

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

RUN_SUFFIX=${SLURM_JOB_ID:-manual}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_ttrl_llama32_3b_base_to_instruct_semantic_class_${RUN_SUFFIX}}
export LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/$EXPERIMENT_NAME}
export EOS_DIAGNOSTIC_OUTPUT_DIR=${EOS_DIAGNOSTIC_OUTPUT_DIR:-$PROJECT_ROOT/diagnostics/$EXPERIMENT_NAME}

# The shared launcher's filename is historical; every model-specific value above
# is passed explicitly and the launcher does not edit downloaded model configs.
exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
