#!/usr/bin/env bash
#SBATCH --job-name=opd_st_q17b_q4b_b16
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
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
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Qwen/Qwen3-4B}
CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-}
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}
TRAIN_TEMPLATE=${TRAIN_TEMPLATE:-ttrl}
EOS_MODE=${EOS_MODE:-baseline}
SEMANTIC_EOS_TOKEN_IDS=${SEMANTIC_EOS_TOKEN_IDS:-}
EOS_MAPPING_EPSILON=${EOS_MAPPING_EPSILON:-1e-12}
case "$TRAIN_TEMPLATE" in
    dapo)
        TRAIN_DATASET="$PROJECT_ROOT/datasets/dapo-math-17k.parquet"
        REWARD_FUNCTION_PATH=verl/verl/utils/reward_score/opd_math/__init__.py
        ;;
    ttrl)
        TRAIN_DATASET="$PROJECT_ROOT/datasets/dapo-math-17k-processed.parquet"
        REWARD_FUNCTION_PATH=verl/verl/utils/reward_score/ttrl_math/__init__.py
        ;;
    eopd)
        TRAIN_DATASET="$PROJECT_ROOT/datasets/dapo-math-17k-raw-question.parquet"
        REWARD_FUNCTION_PATH=verl/verl/utils/reward_score/opd_math/__init__.py
        ;;
    *) echo "Unsupported TRAIN_TEMPLATE: $TRAIN_TEMPLATE" >&2; exit 2 ;;
esac
case "$EOS_MODE" in
    baseline|two_stop|teacher_map|semantic_class|canonical) ;;
    *) echo "Unsupported EOS_MODE: $EOS_MODE" >&2; exit 2 ;;
esac
TEST_DATA_DIR="$PROJECT_ROOT/datasets/test_data"
TEST_DATASET="[$TEST_DATA_DIR/AIME25/test.parquet,$TEST_DATA_DIR/AMC23/test.parquet,$TEST_DATA_DIR/AIME24/test.parquet]"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
REWARD_MAX_TOKEN_LEN_PER_GPU=${REWARD_MAX_TOKEN_LEN_PER_GPU:-$PPO_MAX_TOKEN_LEN_PER_GPU}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
CRITIC_WARMUP=${CRITIC_WARMUP:-0}
EOS_DIAGNOSTIC_OUTPUT_DIR=${EOS_DIAGNOSTIC_OUTPUT_DIR:-}
REWARD_MICRO_BATCH_SIZE=${REWARD_MICRO_BATCH_SIZE:-24}
IS_PLOT=${IS_PLOT:-True}
DRY_RUN=${DRY_RUN:-false}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs/opd_sampled_token_q17b_q4b_b16_6000pro_200step}
RUN_TAG=${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_${TRAIN_TEMPLATE}_${EOS_MODE}_sampled_q17b_q4b_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_${TOTAL_TRAINING_STEPS}step_${RUN_TAG}}
CKPT_PATH="$PROJECT_ROOT/checkpoint/$EXPERIMENT_NAME"
ENABLE_THINKING=${ENABLE_THINKING:-False}
RETURN_MULTI_MODAL_INPUTS=${RETURN_MULTI_MODAL_INPUTS:-True}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-True}
WANDB_MODE=${WANDB_MODE:-online}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-10002}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-19999}

if [ "$DRY_RUN" = "true" ]; then
    echo "project_root=$PROJECT_ROOT"
    echo "actor_model_path=$ACTOR_MODEL_PATH"
    echo "reward_model_path=$REWARD_MODEL_PATH"
    echo "train_template=$TRAIN_TEMPLATE"
    echo "train_dataset=$TRAIN_DATASET"
    echo "reward_function_path=$REWARD_FUNCTION_PATH"
    echo "eos_mode=$EOS_MODE"
    echo "semantic_eos_token_ids=${SEMANTIC_EOS_TOKEN_IDS:-auto}"
    echo "additional_eos_token_ids=manifest-derived"
    echo "chat_template_model=${CHAT_TEMPLATE_MODEL:-student-model}"
    echo "train_batch_size=$TRAIN_BATCH_SIZE"
    echo "rollout_n=$ROLLOUT_N"
    echo "max_response_length=$MAX_RESPONSE_LENGTH"
    echo "ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    echo "reward_max_token_len_per_gpu=$REWARD_MAX_TOKEN_LEN_PER_GPU"
    echo "total_training_steps=$TOTAL_TRAINING_STEPS"
    echo "save_freq=$SAVE_FREQ"
    echo "test_freq=$TEST_FREQ"
    echo "critic_warmup=$CRITIC_WARMUP"
    echo "eos_diagnostic_output_dir=$EOS_DIAGNOSTIC_OUTPUT_DIR"
    echo "enable_thinking=$ENABLE_THINKING"
    echo "return_multi_modal_inputs=$RETURN_MULTI_MODAL_INPUTS"
    echo "enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD"
    echo "wandb_mode=$WANDB_MODE"
    echo "wandb_project=$WANDB_PROJECT"
    echo "ray_worker_port_range=$RAY_MIN_WORKER_PORT-$RAY_MAX_WORKER_PORT"
    echo "log_prob_top_k=0"
    echo "experiment_name=$EXPERIMENT_NAME"
    exit 0
fi

for required_path in "$TRAIN_DATASET"; do
    if [ ! -e "$required_path" ]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$CKPT_PATH"

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
        for cuda_module in "${CUDA_MODULE:-}" cuda/12.9 cuda/12.8 cuda; do
            [ -n "$cuda_module" ] || continue
            if module load "$cuda_module" >/dev/null 2>&1; then
                break
            fi
        done
    fi
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found after CUDA module setup" >&2
    exit 1
fi

export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export CUDA_PATH=${CUDA_PATH:-$CUDA_HOME}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"
export PYTHONPATH="$PROJECT_ROOT/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export TTRL_MATH_VERIFY_WALL_TIMEOUT_SECONDS=2.0
export HF_HOME=${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export WANDB_MODE
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN

RAY_TMPDIR="/tmp/opd_sampled_token_${RUN_TAG}"
TMPDIR="/tmp/opd_sampled_token_tmp_${RUN_TAG}"
export RAY_TMPDIR TMPDIR
export OUTLINES_CACHE_DIR="$PROJECT_ROOT/.cache/outlines/$RUN_TAG"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer_workspace"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$OUTLINES_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE"
source "$PROJECT_ROOT/scripts/ray_job_isolation.sh"

cleanup_ray() {
    opd_cleanup_ray_session
}
trap cleanup_ray EXIT

SEMANTIC_ID_ARGS=()
if [ -n "$SEMANTIC_EOS_TOKEN_IDS" ]; then
    SEMANTIC_ID_ARGS+=(--semantic-eos-token-ids "$SEMANTIC_EOS_TOKEN_IDS")
fi
CHAT_TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE_MODEL" ]; then
    CHAT_TEMPLATE_ARGS+=(--chat-template-model "$CHAT_TEMPLATE_MODEL")
fi
python "$PROJECT_ROOT/scripts/run_semantics.py" write \
    --model "$ACTOR_MODEL_PATH" \
    --teacher-model "$REWARD_MODEL_PATH" \
    --output "$CKPT_PATH/run_semantics.json" \
    --enable-thinking "$ENABLE_THINKING" \
    --eos-mode "$EOS_MODE" \
    --eos-mapping-epsilon "$EOS_MAPPING_EPSILON" \
    --prompt-template "$TRAIN_TEMPLATE" \
    --train-dataset "$TRAIN_DATASET" \
    "${SEMANTIC_ID_ARGS[@]}" \
    "${CHAT_TEMPLATE_ARGS[@]}"

semantics_field() {
    python "$PROJECT_ROOT/scripts/run_semantics.py" get \
        --path "$CKPT_PATH/run_semantics.json" --field "$1"
}
SEMANTIC_EOS_TOKEN_IDS=$(semantics_field semantic_terminal_token_ids)
ADDITIONAL_EOS_TOKEN_IDS=$(semantics_field additional_eos_token_ids)
TRACKED_TOKEN_IDS=$(semantics_field tracked_token_ids)
TRACKED_TOKEN_NAMES=$(semantics_field tracked_token_names)
ENABLE_THINKING=$(semantics_field enable_thinking)

DATA_TEMPLATE_ARGS=()
if [ -n "$CHAT_TEMPLATE_MODEL" ]; then
    DATA_TEMPLATE_ARGS+=("+data.chat_template_model=$CHAT_TEMPLATE_MODEL")
fi

# Ray head start with retries: on busy shared nodes the raylet occasionally misses the GCS
# registration timeout ("node timed out during startup"); a clean retry succeeds.
for ray_attempt in 1 2 3; do
    if ray start --head --temp-dir="$RAY_TMPDIR" --port=0 --dashboard-port=0 \
        --min-worker-port="$RAY_MIN_WORKER_PORT" --max-worker-port="$RAY_MAX_WORKER_PORT" ${RAY_START_EXTRA_ARGS:-}; then
        break
    fi
    echo "ray start attempt $ray_attempt failed; cleaning up and retrying" >&2
    ray stop --force >/dev/null 2>&1 || true
    rm -rf "$RAY_TMPDIR"/session_* "$RAY_TMPDIR"/ray_current_cluster 2>/dev/null || true
    [ "$ray_attempt" = 3 ] && { echo "ray start failed 3 times" >&2; exit 1; }
    sleep $((30 * ray_attempt))
done
opd_export_ray_address
# optional site hook run once the Ray head is up (e.g. cancel redundant Slurm clones); default no-op
${OPDR_AFTER_RAY_START_HOOK:-true}

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs="$RETURN_MULTI_MODAL_INPUTS" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    "${DATA_TEMPLATE_ARGS[@]}" \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.additional_eos_token_ids="[$ADDITIONAL_EOS_TOKEN_IDS]" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload="$ENABLE_ACTIVATION_OFFLOAD" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=fp32 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=0 \
    +actor_rollout_ref.rollout.tracked_token_ids="[$TRACKED_TOKEN_IDS]" \
    +actor_rollout_ref.rollout.tracked_token_names="[$TRACKED_TOKEN_NAMES]" \
    actor_rollout_ref.rollout.eos_mode="$EOS_MODE" \
    actor_rollout_ref.rollout.semantic_eos_token_ids="[$SEMANTIC_EOS_TOKEN_IDS]" \
    actor_rollout_ref.rollout.eos_mapping_epsilon="$EOS_MAPPING_EPSILON" \
    +actor_rollout_ref.rollout.top_k_strategy=only_stu \
    +actor_rollout_ref.rollout.reward_weight_mode=student_p \
    +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.model.path="$REWARD_MODEL_PATH" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.dtype=fp32 \
    reward_model.micro_batch_size_per_gpu="$REWARD_MICRO_BATCH_SIZE" \
    reward_model.forward_max_token_len_per_gpu="$REWARD_MAX_TOKEN_LEN_PER_GPU" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.validation_data_dir="validation_log/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.critic_warmup="$CRITIC_WARMUP" \
    +trainer.eos_diagnostic_output_dir="$EOS_DIAGNOSTIC_OUTPUT_DIR" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot="$IS_PLOT"
