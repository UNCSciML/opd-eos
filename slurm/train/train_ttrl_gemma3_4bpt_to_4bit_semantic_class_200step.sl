#!/usr/bin/env bash
#SBATCH --job-name=opd_g3_4b_sem200
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
# Gemma-3-4B-PT -> Gemma-3-4B-IT TTRL run with the semantic-EOS-class fix.
#
# Run from the repository root:
#   export CONDA_ENV=/path/to/opd-environment
#   hf auth login          # accept the Gemma model licenses first
#   wandb login
#   DRY_RUN=true bash slurm/train/train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl
#   sbatch slurm/train/train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl
#
# What the Gemma baseline run actually shows - read before interpreting this run
# ----------------------------------------------------------------------------
# Gemma's terminal SET is already aligned: gemma-3-4b-pt and gemma-3-4b-it both
# declare [1, 106] = <eos>, <end_of_turn>. two_stop is therefore a no-op here and
# semantic_class does not change rollout stop behaviour, only the loss.
#
# More importantly, semantic_class is NOT expected to repair Gemma's length
# inflation, because there is almost no termination signal for it to aggregate.
# Measured at the positions where the student terminated, over the first 5 steps
# of each baseline run (diagnostics/<run>/step_*_teacher_eos_at_student_eos.npz),
# using the teacher's mass summed over ALL that pair's terminal tokens - the best
# case any surface-form fix can reach:
#
#   pair                          median      mean    share with mass > 0.5
#   Qwen3-1.7B-Base -> Qwen3-4B   8.8e-05     0.221            21.8%
#   Gemma-3-4B-pt   -> 4b-it      9.0e-05     0.013             0.7%
#
# Broken down by where the student stopped, the shapes differ, not just the
# levels (share with teacher mass > 0.5):
#
#   response length   0-64   64-256   256-1024   1024-2048   2048+
#   Qwen                2%      11%        34%         54%     22%
#   Gemma               3%       0%         1%          0%      0%
#
# Qwen's teacher increasingly agrees with the student's stop as the answer gets
# longer, so aggregating the two surface forms recovers a large real signal.
# Gemma is flat near zero at every length: among student stops of 1024 tokens or
# more, 38.2% of Qwen's have teacher mass above 0.5 versus 0 of Gemma's 79.
# Aggregating two near-zero probabilities stays near zero.
#
# Two explanations are ruled out by data in this repository:
#   - The teacher can terminate. In analysis/teacher_rollout_lengths/, gemma-3-4b-it
#     stops on 99.95% of 12800 rollouts of this exact prompt stream, always on
#     token 106, at median length 1601 - the most reliable terminator of the three
#     teachers and the tightest length distribution (std 601 vs Qwen 1485).
#   - The teacher forward pass is not degenerate. At step 1 the Gemma run logs
#     actor/entropy 2.23, teacher/entropy 2.12 and critic/advantages/mean -0.42,
#     against 2.68 / 2.53 / -0.53 for the Qwen baseline. Ordinary-token scoring is
#     comparable; the anomaly is specific to EOS.
#
# So gemma-3-4b-it terminates reliably on its own trajectories yet assigns almost
# no termination probability on the student's trajectories at any length. That is
# a distribution-shift failure, not a surface-token failure, and it is a different
# mechanism from the Qwen EOS mismatch. Run this condition as a control that
# separates the two mechanisms, not as a fix expected to work.
#
#
# This was verified, not assumed
# ------------------------------
# scripts/diagnostic/verify_teacher_eos_scoring.py holds the prompt and the
# scoring code fixed and varies only who wrote the completion. Teacher
# probability of the terminal class, at the position predicting the final token:
#
#   pair    teacher @ its OWN stop   @ random interior   teacher @ STUDENT's stop
#           median   share >0.5      median              median   share >0.5
#   qwen     0.993       91.5%       3.6e-15             1.3e-04      21.9%
#   gemma    1.000      100.0%       2.4e-16             2.0e-04       2.5%
#   llama    0.817       62.4%       2.9e-07             2.1e-03       6.8%
#
# The Gemma teacher is the MOST confident of the three about its own stops
# (median 1.000, 100% above 0.5), so the scoring path is sound and the
# RETURN_MULTI_MODAL_INPUTS=False workaround does not break teacher scoring. The
# interior control is ~0 everywhere, so the metric discriminates. And the Qwen
# arm independently reproduces the in-training diagnostic (21.9% here against
# 21.8% from the training NPZs), so both measurements agree.
#
# The Gemma teacher therefore terminates with near-certainty on its own
# trajectories while assigning ~0 termination probability on the student's. That
# is a genuine distribution-shift result, not an artifact.
#
#
# The cause: the student never writes a finished turn
# ---------------------------------------------------
# The teacher is well calibrated, not merely picky. It scores its own completed
# answers at median 1.000 and a random mid-answer position at 2.4e-16, so it
# gives a stop signal exactly when an answer is finished.
#
# What differs is the student. Counting completions that actually END with a
# boxed answer (requiring the boxed value to be last, since base students often
# restate the prompt's own instruction to box the answer):
#
#   student                  ends with a boxed answer   teacher mass on those
#   Qwen3-1.7B-Base            16 / 118  (13.6%)        median 0.999, 14/16 >0.5
#   gemma-3-4b-pt               0 / 112  ( 0.0%)        no such completion exists
#
# Both students are equally wrong on the task (Qwen 0.8% and Gemma 0.0% correct),
# so this is about the form of the output, not its correctness, and not model
# capacity either - the Gemma student is the larger of the two. Qwen3-1.7B-Base
# already emits instruct-style formatted solutions that Qwen3-4B recognises as a
# finished turn; gemma-3-4b-pt emits code fragments, LaTeX noise, multilingual
# drift and restatements of the prompt.
#
# The practical consequence is that this pair is a poor testbed for an EOS fix.
# OPD can only learn termination when the teacher sometimes signals it where the
# student stops - roughly 14% of the time for Qwen, never for Gemma. Any
# EOS-level fix needs an existing signal to amplify, and here there is none.
#
# teacher_map and canonical are unavailable here: they require exactly one
# student-native EOS id and Gemma has two. They fail preflight explicitly.
#
# Adapting this script to another cluster
# ---------------------------------------
# The #SBATCH header above targets four RTX Pro 6000 GPUs on the original
# cluster. Nothing else in this file is site-specific.
#
# 1. Slurm resources. Override them at submit time rather than editing here:
#      sbatch -p YOUR_PARTITION -A YOUR_ACCOUNT --qos YOUR_QOS \
#        --gres=gpu:4 --cpus-per-task=16 --mem=296G --time=1-00:00:00 \
#        --export=ALL slurm/train/train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl
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
export EOS_MODE=semantic_class
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
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_ttrl_gemma3_4bpt_to_4bit_semantic_class_${RUN_SUFFIX}}
export LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/$EXPERIMENT_NAME}
export EOS_DIAGNOSTIC_OUTPUT_DIR=${EOS_DIAGNOSTIC_OUTPUT_DIR:-$PROJECT_ROOT/diagnostics/$EXPERIMENT_NAME}

# The shared launcher's filename is historical; every model-specific value above
# is passed explicitly and the launcher does not edit downloaded model configs.
exec bash "$PROJECT_ROOT/run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl"
