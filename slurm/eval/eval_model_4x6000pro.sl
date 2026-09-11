#!/usr/bin/env bash
#SBATCH --job-name=opd_model_eval
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
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
MODEL_PATH=${MODEL_PATH:-${1:-}}
EOS_MODE=${EOS_MODE:-baseline}
SEMANTIC_EOS_TOKEN_IDS=${SEMANTIC_EOS_TOKEN_IDS:-151643,151645}
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-}
STOP_TOKEN_IDS=${STOP_TOKEN_IDS-__AUTO__}
BLOCKED_TOKEN_IDS=${BLOCKED_TOKEN_IDS-__AUTO__}
ENABLE_THINKING=${ENABLE_THINKING:-false}
EVAL_TEMPLATE=${EVAL_TEMPLATE:-ttrl}
EVAL_TASKS=${EVAL_TASKS:-AIME24,AIME25,AMC23}
EVAL_N=${EVAL_N:-16}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-8192}
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-12288}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_SEED=${EVAL_SEED:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}
WANDB_GROUP=${WANDB_GROUP:-opd_direct_model_eval}
LOG_WANDB=${LOG_WANDB:-true}
DRY_RUN=${DRY_RUN:-false}

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required: MODEL_PATH=/path/to/hf_model sbatch $0" >&2
    exit 2
fi
case "$EOS_MODE" in
    baseline)
        DEFAULT_STOP_TOKEN_IDS=
        DEFAULT_BLOCKED_TOKEN_IDS=
        ;;
    two_stop)
        DEFAULT_STOP_TOKEN_IDS="$SEMANTIC_EOS_TOKEN_IDS"
        DEFAULT_BLOCKED_TOKEN_IDS=
        ;;
    teacher_map)
        IFS=',' read -r E1 E2 EXTRA_EOS <<< "$SEMANTIC_EOS_TOKEN_IDS"
        if [ -z "${E1:-}" ] || [ -z "${E2:-}" ] || [ -n "${EXTRA_EOS:-}" ]; then
            echo "teacher_map requires exactly two comma-separated semantic EOS ids" >&2
            exit 2
        fi
        DEFAULT_STOP_TOKEN_IDS="$E1"
        DEFAULT_BLOCKED_TOKEN_IDS=
        ;;
    semantic_class)
        DEFAULT_STOP_TOKEN_IDS="$SEMANTIC_EOS_TOKEN_IDS"
        DEFAULT_BLOCKED_TOKEN_IDS=
        ;;
    canonical)
        IFS=',' read -r E1 E2 EXTRA_EOS <<< "$SEMANTIC_EOS_TOKEN_IDS"
        if [ -z "${E1:-}" ] || [ -z "${E2:-}" ] || [ -n "${EXTRA_EOS:-}" ]; then
            echo "canonical requires exactly two comma-separated semantic EOS ids" >&2
            exit 2
        fi
        DEFAULT_STOP_TOKEN_IDS="$E1"
        DEFAULT_BLOCKED_TOKEN_IDS="$E2"
        ;;
    *)
        echo "Unsupported EOS_MODE: $EOS_MODE" >&2
        exit 2
        ;;
esac

if [ "$STOP_TOKEN_IDS" = __AUTO__ ]; then
    STOP_TOKEN_IDS=$DEFAULT_STOP_TOKEN_IDS
fi
if [ "$BLOCKED_TOKEN_IDS" = __AUTO__ ]; then
    BLOCKED_TOKEN_IDS=$DEFAULT_BLOCKED_TOKEN_IDS
fi
case "$ENABLE_THINKING" in
    true|false) ;;
    *) echo "ENABLE_THINKING must be true or false" >&2; exit 2 ;;
esac
case "$EVAL_TEMPLATE" in
    dapo|ttrl|eopd) ;;
    *) echo "Unsupported EVAL_TEMPLATE: $EVAL_TEMPLATE" >&2; exit 2 ;;
esac

MODEL_LABEL=${MODEL_LABEL:-$(basename "$MODEL_PATH")}
RUN_TAG=${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_NAME=${RUN_NAME:-eval_${MODEL_LABEL}_${EVAL_TEMPLATE}_${EOS_MODE}_n${EVAL_N}_${RUN_TAG}}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/eval_outputs/$RUN_NAME}
MERGED_ROOT=${MERGED_ROOT:-$OUTPUT_ROOT/.unused_merged}
DISPLAY_STOP_TOKEN_IDS=${STOP_TOKEN_IDS:-model-default}

print_config() {
    echo "model_path=$MODEL_PATH"
    echo "model_label=$MODEL_LABEL"
    echo "eos_mode=$EOS_MODE"
    echo "semantic_eos_token_ids=$SEMANTIC_EOS_TOKEN_IDS"
    echo "stop_token_ids=$DISPLAY_STOP_TOKEN_IDS"
    echo "blocked_token_ids=$BLOCKED_TOKEN_IDS"
    echo "enable_thinking=$ENABLE_THINKING"
    echo "chat_template_model=${CHAT_TEMPLATE_MODEL:-model-default}"
    echo "eval_template=$EVAL_TEMPLATE"
    echo "tasks=$EVAL_TASKS"
    echo "n=$EVAL_N"
    echo "max_tokens=$EVAL_MAX_TOKENS"
    echo "max_model_len=$EVAL_MAX_MODEL_LEN"
    echo "temperature=$EVAL_TEMPERATURE"
    echo "top_p=$EVAL_TOP_P"
    echo "batched_n_sampling=true"
    echo "output_root=$OUTPUT_ROOT"
    echo "wandb_mode=$WANDB_MODE"
    echo "wandb_project=$WANDB_PROJECT"
}

print_config
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/wandb" "$OUTPUT_ROOT" "$MERGED_ROOT"
cd "$PROJECT_ROOT"

if [ -n "$CONDA_ENV" ]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    elif [ -x "$CONDA_ENV/bin/python" ]; then
        export PATH="$CONDA_ENV/bin:$PATH"
    else
        echo "CONDA_ENV does not contain a usable Python: $CONDA_ENV" >&2
        exit 1
    fi
fi
command -v python >/dev/null 2>&1 || { echo "python not found; activate the OPD environment or set CONDA_ENV" >&2; exit 1; }
hash -r

if ! command -v nvcc >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load "${CUDA_MODULE:-cuda/12.9}" >/dev/null 2>&1 || true
    fi
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
export PYTHONPATH="$PROJECT_ROOT/verl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_MODE
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export TMPDIR=/tmp/opd_model_eval_${RUN_TAG}
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer_workspace"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = true ]; then
    THINKING_ARGS+=(--enable-thinking)
fi
CHAT_TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE_MODEL" ]; then
    CHAT_TEMPLATE_ARGS+=(--chat-template-model "$CHAT_TEMPLATE_MODEL")
fi
WANDB_ARGS=()
if [ "$LOG_WANDB" = true ]; then
    WANDB_ARGS+=(
        --log-wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-group "$WANDB_GROUP"
        --wandb-run-name "$RUN_NAME"
    )
fi

cd "$PROJECT_ROOT/scripts/val/eval"
python opd_eval_curve.py \
    --checkpoint-root "$MODEL_PATH" \
    --base-model "$MODEL_PATH" \
    --merged-root "$MERGED_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --data-dir "$PROJECT_ROOT/scripts/val/data" \
    --tasks "$EVAL_TASKS" \
    --steps 0 \
    --n "$EVAL_N" \
    --max-tokens "$EVAL_MAX_TOKENS" \
    --max-model-len "$EVAL_MAX_MODEL_LEN" \
    --temperature "$EVAL_TEMPERATURE" \
    --top-p "$EVAL_TOP_P" \
    --seed "$EVAL_SEED" \
    --gpu-ids auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --length-tokenizer "$MODEL_PATH" \
    --stop-token-ids "$STOP_TOKEN_IDS" \
    --blocked-token-ids "$BLOCKED_TOKEN_IDS" \
    --prompt-template "$EVAL_TEMPLATE" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${THINKING_ARGS[@]}" \
    --trust-remote-code \
    --batched-n-sampling \
    "${WANDB_ARGS[@]}"
