#!/usr/bin/env bash
#SBATCH --job-name=opd_future_term
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
# Measure the future term that current-token OPD drops, across a sequence of
# checkpoints from one run. Read-only: trains nothing, writes no checkpoints and
# changes no training code.
#
#   DRY_RUN=true bash slurm/diagnostic/measure_future_term_1gpu.sl
#   sbatch --export=ALL slurm/diagnostic/measure_future_term_1gpu.sl
#
# CKPT_ROOT   directory holding step_XXXX/ subdirectories (required)
# STEPS       comma-separated step numbers to measure
# TEACHER     teacher model id or path
# Memory note: --mem=64G is deliberately modest -- it is what a single 4B
# generation plus two scoring passes actually needs. Raise it if your nodes
# schedule on total rather than free memory.

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

CKPT_ROOT=${CKPT_ROOT:-$PROJECT_ROOT/checkpoint_remote/gemma_semantic/checkpoints}
TEACHER=${TEACHER:-google/gemma-3-4b-it}
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$TEACHER}
STEPS=${STEPS:-0020,0040,0060,0080,0200}
RUN_TAG=${RUN_TAG:-gemma_semantic}
NUM_PROMPTS=${NUM_PROMPTS:-64}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_TOKENS=${MAX_TOKENS:-7168}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/analysis/future_term/$RUN_TAG}
CONDA_ENV=${CONDA_ENV:-}
DRY_RUN=${DRY_RUN:-false}

echo "ckpt_root=$CKPT_ROOT"
echo "teacher=$TEACHER"
echo "chat_template_model=$CHAT_TEMPLATE_MODEL"
echo "steps=$STEPS"
echo "num_prompts=$NUM_PROMPTS  n=$ROLLOUT_N  max_tokens=$MAX_TOKENS"
echo "output_dir=$OUTPUT_DIR"

IFS=',' read -r -a STEP_LIST <<< "$STEPS"
for step in "${STEP_LIST[@]}"; do
    [ -d "$CKPT_ROOT/step_$step" ] || { echo "missing $CKPT_ROOT/step_$step" >&2; exit 2; }
done
echo "resolved ${#STEP_LIST[@]} checkpoints"
[ "$DRY_RUN" = true ] && exit 0

if [ -n "$CONDA_ENV" ]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
    elif [ -x "$CONDA_ENV/bin/python" ]; then
        export PATH="$CONDA_ENV/bin:$PATH"
    fi
fi
command -v python >/dev/null 2>&1 || { echo "python not found" >&2; exit 1; }
hash -r
if ! command -v nvcc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/12.9}" >/dev/null 2>&1 || true
fi
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export PYTHONPATH="$PROJECT_ROOT/verl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export TMPDIR=/tmp/opd_future_term_${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUTPUT_DIR"

for step in "${STEP_LIST[@]}"; do
    echo "===== step_$step ====="
    python "$PROJECT_ROOT/scripts/diagnostic/measure_future_term.py" \
        --student-model "$CKPT_ROOT/step_$step" \
        --teacher-model "$TEACHER" \
        --chat-template-model "$CHAT_TEMPLATE_MODEL" \
        --input-parquet "$PROJECT_ROOT/datasets/dapo-math-17k-processed.parquet" \
        --output-dir "$OUTPUT_DIR" \
        --label "step_$step" \
        --num-prompts "$NUM_PROMPTS" \
        --n "$ROLLOUT_N" \
        --max-tokens "$MAX_TOKENS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL"
done
echo "all steps done -> $OUTPUT_DIR"
