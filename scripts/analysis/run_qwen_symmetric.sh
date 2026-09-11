#!/usr/bin/env bash
# Symmetric (question-level) analysis of the Qwen arm: vanilla OPD (A) against
# semantic-EOS OPD (D), each compared to the step-0 base student under the SAME
# generation stop set as the run it is compared with.
#
# Inputs are eval output roots produced by slurm/eval/* (each contains
# grading_summary.json and step_XXXX/*.jsonl + *.tokens.npz):
#
#   BASE_E1     step-0 Qwen3-1.7B-Base eval with EOS_MODE=baseline   (stop {e1})
#   BASE_E1E2   step-0 Qwen3-1.7B-Base eval with EOS_MODE=two_stop   (stop {e1,e2})
#   COND_A      the vanilla-OPD run's eval root
#   COND_D      the semantic-EOS run's eval root
#   OUT         output directory (default: analysis_out/symmetric)
#
# Produce the two base evals with, for example:
#   MODEL_PATH=Qwen/Qwen3-1.7B-Base EOS_MODE=baseline EVAL_TEMPLATE=ttrl \
#     sbatch --export=ALL slurm/eval/eval_model_4x6000pro.sl
#   MODEL_PATH=Qwen/Qwen3-1.7B-Base EOS_MODE=two_stop EVAL_TEMPLATE=ttrl \
#     sbatch --export=ALL slurm/eval/eval_model_4x6000pro.sl
set -euo pipefail

: "${BASE_E1:?set BASE_E1 to the step-0 base eval root with stop {e1}}"
: "${BASE_E1E2:?set BASE_E1E2 to the step-0 base eval root with stop {e1,e2}}"
: "${COND_A:?set COND_A to the vanilla-OPD eval root}"
: "${COND_D:?set COND_D to the semantic-EOS eval root}"
OUT=${OUT:-analysis_out/symmetric}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "$OUT"

# Matched base: each condition is scored against the base student decoded with that
# condition's own stop set, so the comparison isolates the fix rather than the stop set.
"$PYTHON_BIN" scripts/analysis/symmetric_analysis.py \
    --base "$BASE_E1" \
    --cond "A=$COND_A@$BASE_E1" \
    --cond "D=$COND_D@$BASE_E1E2" \
    --out "$OUT/qwen_matchedbase"

# Sensitivity: both conditions against the single stop-{e1} base.
"$PYTHON_BIN" scripts/analysis/symmetric_analysis.py \
    --base "$BASE_E1" \
    --cond "A=$COND_A" \
    --cond "D=$COND_D" \
    --out "$OUT/qwen_e1base"

"$PYTHON_BIN" scripts/analysis/make_symmetric_report.py \
    --run "Qwen3-1.7B-Base <- Qwen3-4B, matched base (A vs base stop{e1}; D vs base stop{e1,e2})=$OUT/qwen_matchedbase" \
    --run "Sensitivity: both A and D vs base stop{e1}=$OUT/qwen_e1base" \
    --base-summary "Qwen3-1.7B-Base, TTRL, stop{e1}=$BASE_E1/grading_summary.json" \
    --base-summary "Qwen3-1.7B-Base, TTRL, stop{e1,e2}=$BASE_E1E2/grading_summary.json" \
    ${NOTES:+--notes "$NOTES"} \
    --out "$OUT/SYMMETRIC_ANALYSIS.md"

echo "report written: $OUT/SYMMETRIC_ANALYSIS.md"
