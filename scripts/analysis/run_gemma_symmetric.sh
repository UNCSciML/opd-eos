#!/usr/bin/env bash
# Symmetric (question-level) analysis of the Gemma arm: vanilla OPD (A) against
# semantic-EOS OPD (D). Gemma's base and instruct models declare the same terminal
# set, so a single base eval serves both conditions.
#
#   BASE      step-0 gemma-3-4b-pt eval root (EOS_MODE=baseline)
#   COND_A    the vanilla-OPD run's eval root
#   COND_D    the semantic-EOS run's eval root
#   OUT       output directory (default: analysis_out/symmetric)
set -euo pipefail

: "${BASE:?set BASE to the step-0 gemma-3-4b-pt eval root}"
: "${COND_A:?set COND_A to the vanilla-OPD eval root}"
: "${COND_D:?set COND_D to the semantic-EOS eval root}"
OUT=${OUT:-analysis_out/symmetric}
TOKENIZER=${TOKENIZER:-google/gemma-3-4b-pt}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "$OUT"

"$PYTHON_BIN" scripts/analysis/symmetric_analysis.py \
    --base "$BASE" \
    --cond "A=$COND_A" \
    --cond "D=$COND_D" \
    --tokenizer "$TOKENIZER" \
    --out "$OUT/gemma_matchedbase"

"$PYTHON_BIN" scripts/analysis/make_symmetric_report.py \
    --run "Gemma-3-4B-PT <- Gemma-3-4B-IT (A = vanilla, D = semantic-EOS; base = gemma-3-4b-pt, stop {1,106})=$OUT/gemma_matchedbase" \
    --base-summary "gemma-3-4b-pt, TTRL, stop{1,106}=$BASE/grading_summary.json" \
    ${NOTES:+--notes "$NOTES"} \
    --out "$OUT/SYMMETRIC_ANALYSIS_GEMMA.md"

echo "report written: $OUT/SYMMETRIC_ANALYSIS_GEMMA.md"
