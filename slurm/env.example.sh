#!/usr/bin/env bash

# Optional: leave empty when the environment is already active before sbatch.
export CONDA_ENV=""

# Hugging Face model IDs work on networked nodes. Replace these with local model
# directories on offline clusters.
export ACTOR_MODEL_PATH="Qwen/Qwen3-1.7B-Base"
export REWARD_MODEL_PATH="Qwen/Qwen3-4B"

# Optional cache and CUDA-module overrides.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

export WANDB_PROJECT="OPD Length Inflation"
export WANDB_MODE="online"
