#!/usr/bin/env bash
# Submit the Llama-3.2-3B and Gemma-3-4B OPD runs and their checkpoint evals.
#
# DISK REQUIREMENTS - READ FIRST
# ------------------------------
# Each 200-step run keeps 10 full training checkpoints (weights plus optimizer
# state). Measured on the completed Gemma-3-4B run: 48 GB per checkpoint, so
# 475 GB per Gemma run; Llama-3.2-3B is proportionally smaller, roughly 350 GB.
# Each eval then merges those checkpoints to HF format, adding about 80 GB
# (Gemma) or 60 GB (Llama) under checkpoint_merged/. Generations are under 1 GB
# per curve.
#
# The default selection below is all four model/EOS-mode combinations, which
# needs roughly 1.9 TB. Confirm the space before submitting, export MERGED_ROOT
# to put merged models on another filesystem, and narrow MODELS/EOS_MODES if you
# cannot hold it all at once.
#
# What this file does
# -------------------
# - Submits one train job per (model, EOS mode) pair selected below.
# - Chains that pair's checkpoint eval with afterok:<train-job-id>, so the eval
#   starts only after its own train job succeeds and already receives the right
#   checkpoint root. No eval semantics are chosen here: each eval inherits them
#   from its train run's run_semantics.json.
# - Prints every job id, dependency edge, experiment name and checkpoint path.
#   It never runs train or eval inside the submitting/login process.
#
# Default plan: 4 jobs per model family pair below, i.e. 2 train + 2 eval for
# llama and the same for gemma. Always inspect it with DRY_RUN first.
#
# Usage
# -----
#   DRY_RUN=true bash slurm/submit_multimodel_runs.sh        # print the plan
#   bash slurm/submit_multimodel_runs.sh                     # submit it
#   MODELS=llama bash slurm/submit_multimodel_runs.sh        # one model family
#   EOS_MODES=semantic_class bash slurm/submit_multimodel_runs.sh
#   RUN_TAG=multimodel-v1 bash slurm/submit_multimodel_runs.sh
#
# MODELS accepts a comma-separated subset of: llama,gemma
# EOS_MODES accepts a comma-separated subset of: baseline,semantic_class
# RUN_TAG is appended to every experiment name; use a stable label so a rerun
# does not collide with an earlier sweep.
#
# What semantic_class means for each pair
# ---------------------------------------
# llama: Llama-3.2-3B declares one terminal token (128001) while the Instruct
#   teacher declares three (128001, 128008, 128009) and ends turns with
#   <|eot_id|>. This is a real student/teacher EOS mismatch and semantic_class
#   is a genuine fix for it.
# gemma: the terminal SET already matches ([1, 106]), so two_stop is a no-op and
#   semantic_class changes only the loss. It is NOT expected to fix Gemma's
#   inflation: measured over the first 5 steps of each baseline run, the teacher's
#   mass summed over all terminal tokens at student stop positions exceeds 0.5 in
#   21.8% of Qwen's terminations but only 0.7% of Gemma's, and among student stops
#   of 1024+ tokens it is 38.2% for Qwen versus 0 of 79 for Gemma. Aggregating two
#   near-zero probabilities stays near zero. gemma-3-4b-it terminates on 99.95% of
#   its own rollouts and its ordinary-token scoring is healthy, so this is a
#   distribution-shift failure rather than a surface-token one - a different
#   mechanism from the Qwen mismatch. This was verified by holding the prompt and
#   the scoring code fixed and varying only who wrote the completion: the Gemma
#   teacher scores its OWN stops at median 1.000 (100% above 0.5) yet the
#   student's at median 2.0e-04 (2.5% above 0.5). The cause is the student's
#   output form: completions that actually end with a boxed answer number 16/118
#   for the Qwen student and 0/112 for the Gemma one, so the Gemma teacher is
#   never shown a finished turn to sign off on. Both students are equally wrong
#   on the task and the Gemma one is the larger, so this is form, not capacity.
#   Run it as a control, not as a fix expected to work. The Gemma train wrapper
#   carries the full measurement.
# teacher_map and canonical are pairwise-only and are unavailable for both of
# these pairs; they fail preflight with an explicit error, so they are not
# offered here.
#
# Adapting this script to another cluster
# ---------------------------------------
# 1. Slurm resources live in the #SBATCH headers of the train/eval entrypoints
#    this script submits, and each of those files documents how to override
#    them. To apply the same override to every job submitted here, export the
#    Slurm environment variables before running this script:
#      export SBATCH_PARTITION=my_gpu_partition
#      export SBATCH_ACCOUNT=my_account
#      export SBATCH_QOS=my_qos
#      export SBATCH_GRES=gpu:4
#      export SBATCH_CPUS_PER_TASK=16
#      export SBATCH_TIMELIMIT=1-00:00:00
#    Keep four GPUs: checkpoints are saved as model_world_size_4_rank_*.pt and
#    the evaluator expects that layout. Leave SBATCH_MEM unset unless you must
#    override it, because train and eval want different amounts.
# 2. Python environment: export CONDA_ENV=/path/to/opd-environment. It is passed
#    through to every job by --export=ALL.
# 3. Checkout location: run this script from anywhere; it resolves PROJECT_ROOT
#    from its own path and exports it to each job.
# 4. Authentication: run `hf auth login` (both model families are gated) and
#    `wandb login`, or export WANDB_MODE=offline and sync afterwards.

set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DRY_RUN=${DRY_RUN:-false}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}
MODELS=${MODELS:-llama,gemma}
EOS_MODES=${EOS_MODES:-baseline,semantic_class}

TRAIN_PREFIX_llama=opd_ttrl_llama32_3b_base_to_instruct
TRAIN_PREFIX_gemma=opd_ttrl_gemma3_4bpt_to_4bit
TRAIN_FILE_STEM_llama=train_ttrl_llama32_3b_base_to_instruct
TRAIN_FILE_STEM_gemma=train_ttrl_gemma3_4bpt_to_4bit
EVAL_FILE_llama=eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl
EVAL_FILE_gemma=eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl

selected() {
    case ",$2," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

submit_one() {
    local model=$1
    local mode=$2
    local prefix train_stem train_script eval_script experiment_name checkpoint_root
    local train_job eval_job

    eval "prefix=\$TRAIN_PREFIX_$model"
    eval "train_stem=\$TRAIN_FILE_STEM_$model"
    eval "eval_script=\$EVAL_FILE_$model"
    train_script="${train_stem}_${mode}_200step.sl"
    experiment_name="${prefix}_${mode}_${RUN_TAG}"
    checkpoint_root="$PROJECT_ROOT/checkpoint/$experiment_name"

    [ -f "$PROJECT_ROOT/slurm/train/$train_script" ] || {
        echo "missing train entrypoint: slurm/train/$train_script" >&2
        exit 1
    }
    [ -f "$PROJECT_ROOT/slurm/eval/$eval_script" ] || {
        echo "missing eval entrypoint: slurm/eval/$eval_script" >&2
        exit 1
    }

    if [ "$DRY_RUN" = true ]; then
        train_job="DRY_RUN-$model-$mode"
        eval_job=DRY_RUN
    else
        train_job=$(
            PROJECT_ROOT="$PROJECT_ROOT" EXPERIMENT_NAME="$experiment_name" WANDB_PROJECT="$WANDB_PROJECT" \
                sbatch --parsable --export=ALL "$PROJECT_ROOT/slurm/train/$train_script"
        )
        eval_job=$(
            PROJECT_ROOT="$PROJECT_ROOT" CHECKPOINT_ROOT="$checkpoint_root" WANDB_PROJECT="$WANDB_PROJECT" \
                sbatch --parsable --export=ALL --dependency="afterok:$train_job" \
                "$PROJECT_ROOT/slurm/eval/$eval_script"
        )
    fi

    echo "model=$model mode=$mode train_job=$train_job experiment_name=$experiment_name"
    echo "model=$model mode=$mode eval_job=$eval_job dependency=afterok:$train_job checkpoint_root=$checkpoint_root"
}

matched=false
for model in llama gemma; do
    selected "$model" "$MODELS" || continue
    for mode in baseline semantic_class; do
        selected "$mode" "$EOS_MODES" || continue
        matched=true
        submit_one "$model" "$mode"
    done
done

if [ "$matched" = false ]; then
    echo "Nothing selected. MODELS must be a subset of llama,gemma and EOS_MODES a subset of baseline,semantic_class" >&2
    exit 2
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete: no jobs were submitted."
fi
