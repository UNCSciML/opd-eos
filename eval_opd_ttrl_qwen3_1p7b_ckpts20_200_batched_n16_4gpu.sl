#!/usr/bin/env bash
#SBATCH --job-name=opd_rt_eval_n16
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
set -euo pipefail
set -x

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
CONDA_ENV=${CONDA_ENV:-}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${1:-}}
if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "Usage: CHECKPOINT_ROOT=/path/to/run sbatch $0" >&2
    exit 2
fi
CHECKPOINT_ROOT=${CHECKPOINT_ROOT%/}
TRAIN_RUN=${TRAIN_RUN:-${CHECKPOINT_ROOT##*/}}
BASE_MODEL=${BASE_MODEL:-}
RUN_SEMANTICS=${RUN_SEMANTICS:-$CHECKPOINT_ROOT/run_semantics.json}
EVAL_TASKS=${EVAL_TASKS:-AIME24,AIME25,AMC23}
EVAL_N=${EVAL_N:-16}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-8192}
EVAL_MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN:-12288}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_SEED=${EVAL_SEED:-0}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}
WANDB_GROUP=${WANDB_GROUP:-opd_length_inflation_checkpoint_eval}
WANDB_MODE=${WANDB_MODE:-online}
DRY_RUN=${DRY_RUN:-false}
OUTPUT_ROOT=${OUTPUT_ROOT:-}
MERGED_ROOT=${MERGED_ROOT:-"$PROJECT_ROOT/checkpoint_merged/${TRAIN_RUN}"}
LOG_DIR="$PROJECT_ROOT/logs/opd_ttrl_q17b_ckpts20_200_eval_n16_4gpu"
STEPS=${STEPS:-20,40,60,80,100,120,140,160,180,200}

if [ ! -f "$RUN_SEMANTICS" ]; then
    echo "Missing train/eval semantics manifest: $RUN_SEMANTICS" >&2
    exit 1
fi
if [ -n "$CONDA_ENV" ] && [ -x "$CONDA_ENV/bin/python" ]; then
    SEMANTICS_PYTHON="$CONDA_ENV/bin/python"
else
    SEMANTICS_PYTHON=$(command -v python || true)
fi
if [ -z "$SEMANTICS_PYTHON" ]; then
    echo "python not found; activate the OPD environment or set CONDA_ENV" >&2
    exit 1
fi
if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL=$(
        "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
            --path "$RUN_SEMANTICS" --field model_path
    )
fi
EOS_MODE=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field eos_mode
)
STOP_TOKEN_IDS=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field rollout_stop_token_ids
)
BLOCKED_TOKEN_IDS=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field blocked_token_ids 2>/dev/null || true
)
TRAIN_TEMPLATE=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field prompt_template
)
ENABLE_THINKING=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field enable_thinking
)
CHAT_TEMPLATE_MODEL=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field chat_template_source 2>/dev/null || true
)
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$BASE_MODEL}
EVAL_TEMPLATE=${EVAL_TEMPLATE:-$TRAIN_TEMPLATE}
case "$EVAL_TEMPLATE" in
    dapo|ttrl|eopd) ;;
    *) echo "Unsupported EVAL_TEMPLATE: $EVAL_TEMPLATE" >&2; exit 2 ;;
esac
if [ -z "$OUTPUT_ROOT" ]; then
    OUTPUT_ROOT="$PROJECT_ROOT/eval_outputs/${TRAIN_RUN}_${EVAL_TEMPLATE}_${EOS_MODE}_think-${ENABLE_THINKING}_m${EVAL_MAX_TOKENS}_n${EVAL_N}_4gpu"
fi

print_config() {
    echo "project_root=$PROJECT_ROOT"
    echo "checkpoint_root=$CHECKPOINT_ROOT"
    echo "base_model=$BASE_MODEL"
    echo "run_semantics=$RUN_SEMANTICS"
    echo "eos_mode=$EOS_MODE"
    echo "stop_token_ids=$STOP_TOKEN_IDS"
    echo "blocked_token_ids=$BLOCKED_TOKEN_IDS"
    echo "enable_thinking=$ENABLE_THINKING"
    echo "chat_template_model=$CHAT_TEMPLATE_MODEL"
    echo "eval_template=$EVAL_TEMPLATE"
    echo "tasks=$EVAL_TASKS"
    echo "n=$EVAL_N"
    echo "max_tokens=$EVAL_MAX_TOKENS"
    echo "save_token_ids=true"
    echo "wandb_project=$WANDB_PROJECT"
    echo "output_root=$OUTPUT_ROOT"
}

print_config
if [ "$DRY_RUN" = true ]; then
    exit 0
fi

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/wandb" "$OUTPUT_ROOT" "$MERGED_ROOT"
cd "$PROJECT_ROOT"

IFS=',' read -r -a requested_steps <<< "$STEPS"
for step in "${requested_steps[@]}"; do
    actor_dir="$CHECKPOINT_ROOT/global_step_${step}/actor"
    if [ ! -f "$actor_dir/huggingface/config.json" ]; then
        echo "Missing checkpoint metadata: $actor_dir/huggingface/config.json" >&2
        exit 1
    fi
    for rank in 0 1 2 3; do
        shard="$actor_dir/model_world_size_4_rank_${rank}.pt"
        if [ ! -f "$shard" ]; then
            echo "Missing checkpoint shard: $shard" >&2
            exit 1
        fi
    done
done

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
    echo "nvcc not found; FlashInfer/vLLM JIT requires a CUDA toolkit" >&2
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
export TMPDIR=/tmp/opd_rt_eval_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer_workspace"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"

cd "$PROJECT_ROOT/scripts/val/eval"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS+=(--enable-thinking)
fi

python opd_eval_curve.py \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --base-model "$BASE_MODEL" \
    --merged-root "$MERGED_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --data-dir "$PROJECT_ROOT/scripts/val/data" \
    --tasks "$EVAL_TASKS" \
    --steps "$STEPS" \
    --n "$EVAL_N" \
    --max-tokens "$EVAL_MAX_TOKENS" \
    --max-model-len "$EVAL_MAX_MODEL_LEN" \
    --temperature "$EVAL_TEMPERATURE" \
    --top-p "$EVAL_TOP_P" \
    --seed "$EVAL_SEED" \
    --gpu-ids auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --length-tokenizer "$BASE_MODEL" \
    --stop-token-ids "$STOP_TOKEN_IDS" \
    --blocked-token-ids "$BLOCKED_TOKEN_IDS" \
    --prompt-template "$EVAL_TEMPLATE" \
    --chat-template-model "$CHAT_TEMPLATE_MODEL" \
    "${THINKING_ARGS[@]}" \
    --trust-remote-code \
    --batched-n-sampling \
    --log-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP" \
    --wandb-run-name "opd_length_inflation_${TRAIN_RUN}_${EVAL_TEMPLATE}_n${EVAL_N}_${SLURM_JOB_ID}"
