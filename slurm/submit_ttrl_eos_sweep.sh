#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DRY_RUN=${DRY_RUN:-false}
SWEEP_TAG=${SWEEP_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
WANDB_PROJECT=${WANDB_PROJECT:-OPD Length Inflation}

submit_condition() {
    local mode=$1
    local train_script=$2
    local experiment_name="opd_ttrl_${mode}_sampled_q17b_q4b_bs16_n4_200step_${SWEEP_TAG}"
    local checkpoint_root="$PROJECT_ROOT/checkpoint/$experiment_name"
    local train_job eval_job

    if [ "$DRY_RUN" = true ]; then
        train_job="DRY_RUN-$mode"
        eval_job=DRY_RUN
    else
        train_job=$(
            PROJECT_ROOT="$PROJECT_ROOT" EXPERIMENT_NAME="$experiment_name" WANDB_PROJECT="$WANDB_PROJECT" \
                sbatch --parsable --export=ALL "$PROJECT_ROOT/slurm/train/$train_script"
        )
        eval_job=$(
            PROJECT_ROOT="$PROJECT_ROOT" CHECKPOINT_ROOT="$checkpoint_root" WANDB_PROJECT="$WANDB_PROJECT" \
                sbatch --parsable --export=ALL --dependency="afterok:$train_job" \
                "$PROJECT_ROOT/slurm/eval/eval_checkpoints_4x6000pro.sl"
        )
    fi

    echo "mode=$mode train_job=$train_job experiment_name=$experiment_name"
    echo "mode=$mode eval_job=$eval_job dependency=afterok:$train_job checkpoint_root=$checkpoint_root"
}

submit_condition baseline train_ttrl_eos_a_baseline_200step.sl
submit_condition two_stop train_ttrl_eos_b_two_stop_200step.sl
submit_condition teacher_map train_ttrl_eos_c_teacher_map_200step.sl
submit_condition semantic_class train_ttrl_eos_d_semantic_class_200step.sl
submit_condition canonical train_ttrl_eos_e_canonical_200step.sl
