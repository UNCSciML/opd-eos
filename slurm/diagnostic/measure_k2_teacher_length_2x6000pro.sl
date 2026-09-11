#!/usr/bin/env bash
#SBATCH --job-name=k2_teacher_len
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2
# K2-Horizon-7B teacher (main) response length / clip rate on the OPD training prompt
# stream, matching the training rollout distribution. This is the reference line for
# the K2 stage figure. Read-only: trains nothing, writes no checkpoints.
#
#   DRY_RUN=true bash slurm/diagnostic/measure_k2_teacher_length_2x6000pro.sl
#   sbatch --export=ALL slurm/diagnostic/measure_k2_teacher_length_2x6000pro.sl
#
# Requires the Llama-view plus the vLLM plugin and HF hook from
# scripts/k2/install_k2_hooks.sh; see docs/K2_LLAMAVIEW_HOWTO.md.
#
# Two GPUs are used as pure data parallelism (tensor_parallel_size=1 each, half the
# prompts per shard), which is what the HOWTO recommends for this 9B-parameter model.
# --mem=96G is deliberate: it is what two 9B vLLM engines need, rather than the
# whole node, so the job schedules on a busy partition.

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT=$(cd -- "$PROJECT_ROOT" && pwd)
elif [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT=$(cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
    PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
fi

MODEL=${MODEL:-$PROJECT_ROOT/checkpoint_remote/k2/K2-Horizon-7B-main-llamaview}
PROMPTS=${PROMPTS:-$PROJECT_ROOT/analysis/teacher_rollout_lengths/k2_prompts_3200.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/analysis/teacher_rollout_lengths/k2_7b_main_ttrl_200step_n4}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_TOKENS=${MAX_TOKENS:-7168}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
GPU_MEM=${GPU_MEM:-0.85}
NUM_SHARDS=${NUM_SHARDS:-2}
CONDA_ENV=${CONDA_ENV:-}
DRY_RUN=${DRY_RUN:-false}

echo "model=$MODEL"
echo "prompts=$PROMPTS ($(wc -l < "$PROMPTS" 2>/dev/null || echo '?') lines)"
echo "n=$ROLLOUT_N max_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN t=$TEMPERATURE top_p=$TOP_P"
echo "shards=$NUM_SHARDS  output_dir=$OUTPUT_DIR"
[ -d "$MODEL" ] || { echo "missing Llama-view: $MODEL" >&2; exit 2; }
[ -f "$PROMPTS" ] || { echo "missing prompts: $PROMPTS" >&2; exit 2; }
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
# fork, not spawn: k2_teacher_length_vllm.py runs at module top level with no
# `if __name__ == "__main__"` guard, and spawn makes the engine-core child re-import
# and re-execute it, which multiprocessing refuses ("attempt to start a new process
# before the current process has finished its bootstrapping phase").
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export TMPDIR=/tmp/k2_teacher_len_${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR" "$OUTPUT_DIR"

# The Llama-view config carries layernorm_num_groups=4; the HF hook only activates on
# this flag, and the vLLM plugin loads through its entry point.
export OPD_K2_GROUPED_RMSNORM=1

python - "$PROMPTS" "$OUTPUT_DIR" "$NUM_SHARDS" <<'PYSPLIT'
import json, sys, pathlib
src, out, k = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
rows = [l for l in src.read_text().splitlines() if l.strip()]
for i in range(k):
    (out / f"shard_{i}.jsonl").write_text("\n".join(rows[i::k]) + "\n")
print(f"split {len(rows)} prompts into {k} shards", flush=True)
PYSPLIT

# Smoke the engine on a handful of prompts before committing two GPUs for hours: a
# plugin/template/dtype problem shows up here in a couple of minutes.
if [ "${SMOKE:-true}" = true ]; then
    echo "===== smoke (4 prompts, 512 tokens) ====="
    CUDA_VISIBLE_DEVICES=0 python "$PROJECT_ROOT/scripts/k2/k2_teacher_length_vllm.py" \
        --model "$MODEL" --prompts "$PROMPTS" --field prompt --limit 4 \
        --n 2 --max-tokens 512 --max-model-len "$MAX_MODEL_LEN" \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" \
        --tp 1 --gpu-mem "$GPU_MEM" --out "$OUTPUT_DIR/smoke.json" \
        > "$OUTPUT_DIR/smoke.log" 2>&1 || {
            echo "smoke failed" >&2; tail -n 40 "$OUTPUT_DIR/smoke.log" >&2; exit 1; }
    python -c "import json,sys; s=json.load(open('$OUTPUT_DIR/smoke.json'))['summary']; print(json.dumps(s,indent=1)); sys.exit(0 if s['n_samples']==8 else 1)"
    echo "===== smoke OK ====="
fi

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python "$PROJECT_ROOT/scripts/k2/k2_teacher_length_vllm.py" \
        --model "$MODEL" \
        --prompts "$OUTPUT_DIR/shard_$i.jsonl" --field prompt \
        --n "$ROLLOUT_N" --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
        --temperature "$TEMPERATURE" --top-p "$TOP_P" \
        --tp 1 --gpu-mem "$GPU_MEM" \
        --out "$OUTPUT_DIR/shard_$i.json" > "$OUTPUT_DIR/shard_$i.log" 2>&1 &
    pids+=("$!")
    echo "launched shard $i on GPU $i (pid ${pids[-1]})"
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" = 0 ] || { echo "a shard failed; see $OUTPUT_DIR/shard_*.log" >&2; tail -n 20 "$OUTPUT_DIR"/shard_*.log >&2; exit 1; }

python - "$OUTPUT_DIR" "$NUM_SHARDS" "$MODEL" <<'PYMERGE'
import json, pathlib, statistics, sys
out, k, model = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lens, clip, eod, eot, other, npr = [], 0, 0, 0, 0, 0
for i in range(k):
    d = json.load(open(out / f"shard_{i}.json"))
    s = d["summary"]; n = s["n_samples"]; npr += s["n_prompts"]
    lens += [r["len"] for r in d["samples"]]
    clip += round(s["clip_rate"] * n)
    eod += round(s["end_EOD_frac"] * n); eot += round(s["end_EOT_frac"] * n)
    other += round(s["end_other_frac"] * n)
n = len(lens); lens_sorted = sorted(lens)
summary = {
    "model": model, "n_prompts": npr, "n_samples": n,
    "response_length_mean": statistics.mean(lens),
    "response_length_median": statistics.median(lens),
    "response_length_p95": lens_sorted[int(0.95 * (n - 1))],
    "response_length_min": lens_sorted[0], "response_length_max": lens_sorted[-1],
    "response_length_std": statistics.pstdev(lens),
    # clip via finish_reason == "length"; never via len(ids) >= max_tokens, which is
    # unreachable when prompt + max_tokens exceeds max_model_len (see the HOWTO).
    "length_capped_fraction": clip / n,
    "end_EOD_frac": eod / n, "end_EOT_frac": eot / n, "end_other_frac": other / n,
}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
PYMERGE
echo "done -> $OUTPUT_DIR/summary.json"
