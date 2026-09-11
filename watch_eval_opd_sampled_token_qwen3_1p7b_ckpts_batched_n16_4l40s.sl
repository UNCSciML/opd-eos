#!/usr/bin/env bash
#SBATCH --job-name=opd_sampled_watch_eval
#SBATCH --output=opd_sampled_watch_eval_%j.out
#SBATCH --error=opd_sampled_watch_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
CONDA_ENV=${CONDA_ENV:-}
CHECKPOINT_ROOT=
BASE_MODEL=
OUTPUT_ROOT=
MERGED_ROOT=
RUN_SEMANTICS=
LOG_DIR=
POLL_SECONDS=30
FINAL_STEP=200
WORLD_SIZE=4
EVAL_TEMPLATES=
DRY_RUN=false
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}

usage() {
    echo "Usage: $0 --checkpoint-root PATH [options]" >&2
    echo "       --base-model defaults to model_path in run_semantics.json" >&2
    echo "Options: --output-root PATH --merged-root PATH --run-semantics PATH" >&2
    echo "         --conda-env PATH --log-dir PATH --poll-seconds N --final-step N --world-size N" >&2
    echo "         --eval-templates ttrl,dapo,eopd --dry-run" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint-root)
            CHECKPOINT_ROOT=${2:?Missing value for --checkpoint-root}
            shift 2
            ;;
        --base-model)
            BASE_MODEL=${2:?Missing value for --base-model}
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT=${2:?Missing value for --output-root}
            shift 2
            ;;
        --merged-root)
            MERGED_ROOT=${2:?Missing value for --merged-root}
            shift 2
            ;;
        --run-semantics)
            RUN_SEMANTICS=${2:?Missing value for --run-semantics}
            shift 2
            ;;
        --conda-env)
            CONDA_ENV=${2:?Missing value for --conda-env}
            shift 2
            ;;
        --log-dir)
            LOG_DIR=${2:?Missing value for --log-dir}
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS=${2:?Missing value for --poll-seconds}
            shift 2
            ;;
        --final-step)
            FINAL_STEP=${2:?Missing value for --final-step}
            shift 2
            ;;
        --world-size)
            WORLD_SIZE=${2:?Missing value for --world-size}
            shift 2
            ;;
        --eval-templates)
            EVAL_TEMPLATES=${2:?Missing value for --eval-templates}
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [ -z "$CHECKPOINT_ROOT" ]; then
    usage
    exit 2
fi

CHECKPOINT_ROOT=${CHECKPOINT_ROOT%/}
RUN_NAME=${CHECKPOINT_ROOT##*/}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PROJECT_ROOT/eval_outputs/${RUN_NAME}_batched_8192_n16_4l40s_watch"}
MERGED_ROOT=${MERGED_ROOT:-"$PROJECT_ROOT/checkpoint_merged/${RUN_NAME}_4l40s_watch"}
RUN_SEMANTICS=${RUN_SEMANTICS:-"$CHECKPOINT_ROOT/run_semantics.json"}
LOG_DIR=${LOG_DIR:-"$PROJECT_ROOT/logs/${RUN_NAME}_watch_eval_n16_4l40s"}

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Checkpoint root does not exist: $CHECKPOINT_ROOT" >&2
    exit 1
fi
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
SCHEMA_VERSION=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field schema_version
)
if [ "$SCHEMA_VERSION" != "2" ] && [ "$SCHEMA_VERSION" != "3" ]; then
    echo "Unsupported run_semantics schema_version=$SCHEMA_VERSION; expected 2 or 3" >&2
    exit 1
fi
if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL=$(
        "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
            --path "$RUN_SEMANTICS" --field model_path
    )
fi

STOP_TOKEN_IDS=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field rollout_stop_token_ids
)
BLOCKED_TOKEN_IDS=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field blocked_token_ids 2>/dev/null || true
)
ENABLE_THINKING=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field enable_thinking
)
PROMPT_TEMPLATE=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field prompt_template
)
EOS_MODE=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field eos_mode
)
CHAT_TEMPLATE_MODEL=$(
    "$SEMANTICS_PYTHON" "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$RUN_SEMANTICS" --field chat_template_source 2>/dev/null || true
)
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$BASE_MODEL}
if [ -z "$EVAL_TEMPLATES" ]; then
    EVAL_TEMPLATES=$PROMPT_TEMPLATE
fi
IFS=',' read -r -a EVAL_TEMPLATE_LIST <<< "$EVAL_TEMPLATES"
for eval_template in "${EVAL_TEMPLATE_LIST[@]}"; do
    if [ "$eval_template" != "ttrl" ] && [ "$eval_template" != "dapo" ] && [ "$eval_template" != "eopd" ]; then
        echo "Unsupported eval template: $eval_template" >&2
        exit 2
    fi
done
if [ -z "$STOP_TOKEN_IDS" ]; then
    echo "No effective EOS token ids recorded in $RUN_SEMANTICS" >&2
    exit 1
fi
if [ "$ENABLE_THINKING" != "true" ] && [ "$ENABLE_THINKING" != "false" ]; then
    echo "Invalid enable_thinking value in $RUN_SEMANTICS: $ENABLE_THINKING" >&2
    exit 1
fi

checkpoint_ready() {
    local step=$1
    local checkpoint_dir="$CHECKPOINT_ROOT/global_step_${step}"
    local actor_dir="$checkpoint_dir/actor"

    [ -f "$checkpoint_dir/data.pt" ] || return 1
    [ -f "$actor_dir/huggingface/config.json" ] || return 1
    [ -f "$actor_dir/fsdp_config.json" ] || return 1
    local rank
    for ((rank = 0; rank < WORLD_SIZE; rank++)); do
        [ -f "$actor_dir/model_world_size_${WORLD_SIZE}_rank_${rank}.pt" ] || return 1
    done
}

discover_steps() {
    find "$CHECKPOINT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
        | sed -n 's/^global_step_\([0-9][0-9]*\)$/\1/p' \
        | sort -n
}

print_config() {
    echo "checkpoint_root=$CHECKPOINT_ROOT"
    echo "base_model=$BASE_MODEL"
    echo "output_root=$OUTPUT_ROOT"
    echo "merged_root=$MERGED_ROOT"
    echo "run_semantics=$RUN_SEMANTICS"
    echo "enable_thinking=$ENABLE_THINKING"
    echo "stop_token_ids=$STOP_TOKEN_IDS"
    echo "blocked_token_ids=$BLOCKED_TOKEN_IDS"
    echo "prompt_template=$PROMPT_TEMPLATE"
    echo "eos_mode=$EOS_MODE"
    echo "chat_template_model=$CHAT_TEMPLATE_MODEL"
    echo "eval_templates=$EVAL_TEMPLATES"
    echo "final_step=$FINAL_STEP poll_seconds=$POLL_SECONDS world_size=$WORLD_SIZE"
    echo "wandb_project=$WANDB_PROJECT"
}

if [ "$DRY_RUN" = true ]; then
    print_config
    while IFS= read -r step; do
        [ "$step" -le "$FINAL_STEP" ] || continue
        if [ -f "$OUTPUT_ROOT/.completed_step_${step}" ]; then
            echo "step=$step status=completed"
        elif checkpoint_ready "$step"; then
            echo "step=$step status=ready"
        else
            echo "step=$step status=incomplete"
        fi
    done < <(discover_steps)
    exit 0
fi

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/wandb" "$OUTPUT_ROOT" "$MERGED_ROOT"
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
    echo "nvcc not found; FlashInfer/vLLM JIT requires a CUDA toolkit" >&2
    exit 1
fi

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export CUDA_PATH=${CUDA_PATH:-$CUDA_HOME}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}

export PYTHONPATH="$PROJECT_ROOT/verl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export TMPDIR=/tmp/opd_sampled_watch_eval_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer_workspace"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS+=(--enable-thinking)
fi

evaluate_step() {
    local step=$1
    local step_padded
    step_padded=$(printf '%04d' "$step")

    echo "[watcher] evaluating global_step_${step}"
    cd "$PROJECT_ROOT/scripts/val/eval"
    local eval_template template_output
    for eval_template in "${EVAL_TEMPLATE_LIST[@]}"; do
        template_output="$OUTPUT_ROOT/$eval_template"
        mkdir -p "$template_output"
        if ! python opd_eval_curve.py \
            --checkpoint-root "$CHECKPOINT_ROOT" \
            --base-model "$BASE_MODEL" \
            --merged-root "$MERGED_ROOT" \
            --output-root "$template_output" \
            --data-dir "$PROJECT_ROOT/scripts/val/data" \
            --tasks AIME24,AIME25,AMC23 \
            --steps "$step" \
            --n 16 \
            --max-tokens 8192 \
            --max-model-len 12288 \
            --temperature 0.7 \
            --top-p 0.95 \
            --seed 0 \
            --gpu-ids auto \
            --gpu-memory-utilization 0.85 \
            --length-tokenizer "$BASE_MODEL" \
            --stop-token-ids "$STOP_TOKEN_IDS" \
            --blocked-token-ids "$BLOCKED_TOKEN_IDS" \
            --prompt-template "$eval_template" \
            --chat-template-model "$CHAT_TEMPLATE_MODEL" \
            "${THINKING_ARGS[@]}" \
            --trust-remote-code \
            --batched-n-sampling \
            --log-wandb \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-group "opd_length_inflation_${eval_template}_sampled_token_q17b_ckpt_curve_eval" \
            --wandb-run-name "opd_length_inflation_${RUN_NAME}_${eval_template}_step_${step_padded}_n16_${SLURM_JOB_ID}"; then
            echo "[watcher] evaluation failed for global_step_${step} template=${eval_template}" >&2
            return 1
        fi
        cp "$template_output/grading_summary.json" "$template_output/grading_summary_step_${step_padded}.json"
        cp "$template_output/grading_summary.csv" "$template_output/grading_summary_step_${step_padded}.csv"
    done

    touch "$OUTPUT_ROOT/.completed_step_${step}"
    echo "[watcher] completed global_step_${step} templates=${EVAL_TEMPLATES}"
    return 0
}

write_final_summary() {
    local completed_steps
    completed_steps=$(find "$OUTPUT_ROOT" -maxdepth 1 -type f -name '.completed_step_*' -printf '%f\n' \
        | sed -n 's/^\.completed_step_\([0-9][0-9]*\)$/\1/p' \
        | sort -n \
        | paste -sd, -)
    [ -n "$completed_steps" ] || return 0

    echo "[watcher] rebuilding aggregate summaries for steps=$completed_steps"
    cd "$PROJECT_ROOT/scripts/val/eval"
    local eval_template
    for eval_template in "${EVAL_TEMPLATE_LIST[@]}"; do
        python opd_eval_curve.py \
            --checkpoint-root "$CHECKPOINT_ROOT" \
            --base-model "$BASE_MODEL" \
            --merged-root "$MERGED_ROOT" \
            --output-root "$OUTPUT_ROOT/$eval_template" \
            --data-dir "$PROJECT_ROOT/scripts/val/data" \
            --tasks AIME24,AIME25,AMC23 \
            --steps "$completed_steps" \
            --n 16 \
            --max-tokens 8192 \
            --max-model-len 12288 \
            --temperature 0.7 \
            --top-p 0.95 \
            --seed 0 \
            --gpu-ids auto \
            --gpu-memory-utilization 0.85 \
            --length-tokenizer "$BASE_MODEL" \
            --stop-token-ids "$STOP_TOKEN_IDS" \
            --blocked-token-ids "$BLOCKED_TOKEN_IDS" \
            --prompt-template "$eval_template" \
            --chat-template-model "$CHAT_TEMPLATE_MODEL" \
            "${THINKING_ARGS[@]}" \
            --trust-remote-code \
            --batched-n-sampling
    done
}

print_config
echo "[watcher] monitoring $CHECKPOINT_ROOT every ${POLL_SECONDS}s through step ${FINAL_STEP}"
while true; do
    evaluation_failed=0
    while IFS= read -r step; do
        [ "$step" -le "$FINAL_STEP" ] || continue
        [ -f "$OUTPUT_ROOT/.completed_step_${step}" ] && continue
        checkpoint_ready "$step" || continue

        if ! evaluate_step "$step"; then
            evaluation_failed=1
            break
        fi
    done < <(discover_steps)

    if [ -f "$OUTPUT_ROOT/.completed_step_${FINAL_STEP}" ]; then
        write_final_summary
        echo "[watcher] final checkpoint ${FINAL_STEP} evaluated; exiting"
        exit 0
    fi

    if [ "$evaluation_failed" -eq 0 ]; then
        echo "[watcher] waiting for a new complete checkpoint"
    fi
    sleep "$POLL_SECONDS"
done
