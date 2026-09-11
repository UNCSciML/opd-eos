#!/usr/bin/env bash
#SBATCH --job-name=opd_eos_verify
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
# Verify whether the teacher's near-zero EOS probability on student rollouts is a
# scoring artifact or a real distribution-shift result. Holds the prompt and the
# scoring code fixed and varies only who wrote the completion. Read-only: trains
# nothing and writes no checkpoints.
#
#   DRY_RUN=true bash slurm/diagnostic/verify_teacher_eos_scoring_1gpu.sl
#   sbatch --export=ALL slurm/diagnostic/verify_teacher_eos_scoring_1gpu.sl
#
# PAIR=qwen|gemma|llama selects the model pair (default gemma).

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

PAIR=${PAIR:-gemma}
case "$PAIR" in
    gemma) STUDENT=google/gemma-3-4b-pt;       TEACHER=google/gemma-3-4b-it ;;
    qwen)  STUDENT=Qwen/Qwen3-1.7B-Base;       TEACHER=Qwen/Qwen3-4B ;;
    llama) STUDENT=meta-llama/Llama-3.2-3B;    TEACHER=meta-llama/Llama-3.2-3B-Instruct ;;
    *) echo "PAIR must be gemma, qwen or llama" >&2; exit 2 ;;
esac

CONDA_ENV=${CONDA_ENV:-}
NUM_PROMPTS=${NUM_PROMPTS:-64}
EVAL_N=${EVAL_N:-2}
MAX_TOKENS=${MAX_TOKENS:-4096}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/analysis/teacher_eos_scoring_check/$PAIR}
DRY_RUN=${DRY_RUN:-false}

echo "pair=$PAIR"
echo "student=$STUDENT"
echo "teacher=$TEACHER"
echo "num_prompts=$NUM_PROMPTS"
echo "n=$EVAL_N"
echo "max_tokens=$MAX_TOKENS"
echo "output_dir=$OUTPUT_DIR"
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
export TMPDIR=/tmp/opd_eos_verify_${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUTPUT_DIR"

python "$PROJECT_ROOT/scripts/diagnostic/verify_teacher_eos_scoring.py" \
    --student-model "$STUDENT" \
    --teacher-model "$TEACHER" \
    --input-parquet "$PROJECT_ROOT/datasets/dapo-math-17k-processed.parquet" \
    --output-dir "$OUTPUT_DIR" \
    --num-prompts "$NUM_PROMPTS" \
    --n "$EVAL_N" \
    --max-tokens "$MAX_TOKENS"
