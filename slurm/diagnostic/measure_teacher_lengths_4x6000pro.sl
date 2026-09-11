#!/usr/bin/env bash
#SBATCH --job-name=teacher_len_n4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

CONDA_ENV=${CONDA_ENV:-}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
INPUT_PARQUET=${INPUT_PARQUET:-$PROJECT_ROOT/datasets/dapo-math-17k-processed.parquet}
MAX_PROMPTS=${MAX_PROMPTS:-3200}
N=${N:-4}
MAX_TOKENS=${MAX_TOKENS:-7168}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
STOP_TOKEN_IDS=${STOP_TOKEN_IDS:-151645,151643}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_IDS=${GPU_IDS:-auto}
ENABLE_THINKING=${ENABLE_THINKING:-false}
RUN_TAG=${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/analysis/teacher_rollout_lengths/qwen3_4b_ttrl_200step_n4_${RUN_TAG}}
DRY_RUN=${DRY_RUN:-false}

print_config() {
    echo "project_root=$PROJECT_ROOT"
    echo "model_path=$MODEL_PATH"
    echo "input_parquet=$INPUT_PARQUET"
    echo "max_prompts=$MAX_PROMPTS"
    echo "n=$N"
    echo "max_tokens=$MAX_TOKENS"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "temperature=$TEMPERATURE"
    echo "top_p=$TOP_P"
    echo "top_k=$TOP_K"
    echo "repetition_penalty=$REPETITION_PENALTY"
    echo "stop_token_ids=$STOP_TOKEN_IDS"
    echo "enable_thinking=$ENABLE_THINKING"
    echo "tensor_parallel_size=1"
    echo "num_workers=4"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    echo "output_dir=$OUTPUT_DIR"
}

print_config
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

if [ ! -f "$INPUT_PARQUET" ]; then
    echo "Missing input dataset: $INPUT_PARQUET" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing teacher model: $MODEL_PATH" >&2
    exit 1
fi
case "$ENABLE_THINKING" in
    true|false) ;;
    *) echo "ENABLE_THINKING must be true or false" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif [ -x "$CONDA_ENV/bin/python" ]; then
    export PATH="$CONDA_ENV/bin:$PATH"
else
    echo "CONDA_ENV does not contain a usable Python: $CONDA_ENV" >&2
    exit 1
fi
hash -r

if ! command -v nvcc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/12.9}" >/dev/null 2>&1 || true
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found; vLLM/FlashInfer requires a CUDA toolkit" >&2
    exit 1
fi

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export CUDA_PATH=${CUDA_PATH:-$CUDA_HOME}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export TMPDIR=/tmp/teacher_length_${RUN_TAG}
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer_workspace"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = true ]; then
    THINKING_ARGS+=(--enable-thinking)
fi

python "$PROJECT_ROOT/scripts/infer/teacher_length_rollout.py" \
    --input-parquet "$INPUT_PARQUET" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-prompts "$MAX_PROMPTS" \
    --n "$N" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --stop-token-ids "$STOP_TOKEN_IDS" \
    --gpu-ids "$GPU_IDS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    "${THINKING_ARGS[@]}"
