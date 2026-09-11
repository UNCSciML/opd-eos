# Slurm entrypoint reference

This is the detailed reference for the Slurm entrypoints behind the Qwen, Llama,
Gemma and K2 OPD experiments. Start from the top-level [README](../README.md) for
setup and a quickstart; come here for the per-entrypoint contract, the full
formal settings, and the cross-family EOS analysis.

No script depends on a particular checkout, conda environment, model cache or
output path. Run all commands from the repository root.

## 1. Entrypoints

```text
slurm/
├── env.example.sh
├── submit_ttrl_eos_sweep.sh
├── submit_multimodel_runs.sh
├── train/
│   ├── train_ttrl_eos_a_baseline_200step.sl                              # Qwen, EOS condition A
│   ├── train_ttrl_eos_b_two_stop_200step.sl                              #   ... B
│   ├── train_ttrl_eos_c_teacher_map_200step.sl                           #   ... C
│   ├── train_ttrl_eos_d_semantic_class_200step.sl                        #   ... D
│   ├── train_ttrl_eos_e_canonical_200step.sl                             #   ... E
│   ├── train_dapo_baseline_200step.sl                                    # Qwen, DAPO template
│   ├── train_eopd_baseline_200step.sl                                    # Qwen, raw-question template
│   ├── train_ttrl_llama32_3b_base_to_instruct_baseline_200step.sl
│   ├── train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
│   ├── train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl
│   ├── train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl
│   ├── train_ttrl_k2_7b_pretrainfinal_to_main_baseline_400step.sl        # K2 long horizon
│   ├── train_ttrl_k2_7b_pretrainfinal_to_main_semantic_400step.sl        #   ... with the EOS fix
│   ├── train_ttrl_k2_7b_mid1final_to_main_baseline_200step.sl            # K2 stage controls
│   ├── train_ttrl_k2_7b_mid1final_to_main_semantic_200step.sl
│   ├── train_ttrl_k2_7b_sft1final_to_main_baseline_200step.sl
│   └── train_ttrl_k2_7b_sft1final_to_main_semantic_200step.sl
├── diagnostic/
│   ├── verify_teacher_eos_scoring_1gpu.sl
│   ├── diagnose_baseline_teacher_eos_5step_4x6000pro.sl
│   ├── measure_teacher_lengths_4x6000pro.sl
│   ├── measure_k2_teacher_length_2x6000pro.sl
│   └── measure_future_term_1gpu.sl
├── eval/
│   ├── eval_model_4x6000pro.sl                                           # any HF / merged model
│   ├── eval_checkpoints_4x6000pro.sl                                     # Qwen checkpoint curve
│   ├── eval_base_step0_templates_2x6000pro.sl
│   ├── eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl
│   ├── eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl
│   ├── eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl                    # every K2 run
│   └── watch_checkpoints_4xl40s.sl                                       # evaluate a live run
└── smoke/
    └── smoke_eval_base_4x6000pro.sl
```

The train files are small condition-specific wrappers around
`run_opd_sampled_token_qwen3_1p7b_base_4b_bs16_6000pro_200step.sl`. This keeps
one reproducible `.sl` per formal condition while preventing the common Hydra
configuration from drifting between conditions. The launcher's filename is
historical; model-specific wrappers explicitly replace its model and template
defaults.

The two model-specific eval files are the same kind of wrapper around
`eval_opd_ttrl_qwen3_1p7b_ckpts20_200_batched_n16_4gpu.sl`. They add only the
Slurm resources a 3B/4B student needs and a required `CHECKPOINT_ROOT`; every
eval setting still comes from the shared evaluator and the train manifest, so a
Llama or Gemma curve is directly comparable with the Qwen curves.

## 2. Software environment

### 2.1 Requirements

- Linux and Slurm;
- four CUDA GPUs for the supplied formal jobs;
- a CUDA toolkit with `nvcc` visible inside the job;
- Python 3.12, PyTorch, vLLM, VERL, Ray, Transformers, FlashInfer, W&B,
  PyArrow, pandas and the math-grading dependencies installed by VERL.

One stack that has passed both train and eval smoke tests is:

| Package | Tested version |
|---|---:|
| Python | 3.12.13 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| vLLM | 0.11.0 |
| Transformers | 4.55.4 |
| Ray | 2.54.0 |
| FlashInfer | 0.3.1 |
| W&B | 0.28.0 |

Example conda installation (adapt the CUDA wheel to the cluster driver):

```bash
conda create -n opd-length-inflation python=3.12 -y
conda activate opd-length-inflation
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install "vllm==0.11.0" "transformers==4.55.4" \
  "ray==2.54.0" "wandb==0.28.0" "flashinfer-python==0.3.1"
python -m pip install -e ./verl
```

If the cluster already provides a working VERL/vLLM environment, use it
instead. The scripts follow this environment policy:

1. If `CONDA_ENV` is set, activate it (or prepend its `bin` directory).
2. Otherwise use the Python environment inherited by `sbatch`.
3. If `nvcc` is absent and the cluster has Environment Modules, try
   `CUDA_MODULE` and then common CUDA module names.
4. Fail with an explicit error if Python or `nvcc` is still unavailable.

### 2.2 Runtime configuration

Use the example as a template outside the git checkout so local paths are not
accidentally committed:

```bash
mkdir -p "$HOME/.config"
cp slurm/env.example.sh "$HOME/.config/opd-length-inflation-env.sh"
# Edit CONDA_ENV/model/cache values if the compute nodes are offline.
source "$HOME/.config/opd-length-inflation-env.sh"
```

Defaults use Hugging Face IDs and the standard user cache:

```bash
ACTOR_MODEL_PATH=Qwen/Qwen3-1.7B-Base
REWARD_MODEL_PATH=Qwen/Qwen3-4B
HF_HOME=${XDG_CACHE_HOME:-$HOME/.cache}/huggingface
WANDB_PROJECT="OPD Length Inflation"
WANDB_MODE=online
```

On compute nodes without Internet access, set `ACTOR_MODEL_PATH` and
`REWARD_MODEL_PATH` to shared local Hugging Face model directories. The scripts
never edit model `config.json`, `generation_config.json`, or tokenizer files.

Authenticate W&B once before submitting online runs:

```bash
wandb login
```

Llama 3.2 and Gemma 3 are gated Hugging Face repositories. Accept their model
licenses and authenticate before submitting those jobs:

```bash
hf auth login
```

Use `WANDB_MODE=offline` when the compute node has no network. Train and eval
semantics are unchanged.

### 2.3 Cluster-specific Slurm resources

Every `#SBATCH` header requests only a GPU **count** (`--gres=gpu:N`) and names
no partition, so the jobs land on your cluster's default partition. Select the
partition, a specific GPU model, an account or a QOS with command-line `sbatch`
options or `SBATCH_*` environment variables, which override the embedded
headers:

```bash
export SBATCH_PARTITION=my_gpu_partition
export SBATCH_GRES=gpu:h100:4          # a specific GPU model, if your site requires one
export SBATCH_ACCOUNT=my_account
export SBATCH_CPUS_PER_TASK=16
```

The `--mem` and `--time` values in the headers are the ones the runs actually
needed on 4x 96 GiB GPUs (see sections 4.3 and 4.4); treat them as a starting
point rather than a requirement.

Slurm stdout/stderr use portable names such as `%x_%j.out` in the directory
where `sbatch` is invoked. No output path is hard-coded.

Submit from the repository root. Slurm normally runs a copied script from its
spool directory, so every entrypoint resolves the checkout from
`PROJECT_ROOT` first and then from `SLURM_SUBMIT_DIR`; the sweep helper exports
`PROJECT_ROOT` automatically. If jobs are submitted from elsewhere, provide it
explicitly:

```bash
PROJECT_ROOT=/shared/path/opd-length-inflation \
  sbatch --export=ALL,PROJECT_ROOT slurm/train/train_ttrl_eos_a_baseline_200step.sl
```

## 3. Data

Expected repository-relative files:

```text
datasets/dapo-math-17k.parquet                 # DAPO prompt
datasets/dapo-math-17k-processed.parquet       # TTRL prompt
datasets/dapo-math-17k-raw-question.parquet    # EOPD/raw-question prompt
scripts/val/data/AIME24/test.parquet
scripts/val/data/AIME25/test.parquet
scripts/val/data/AMC23/test.parquet
```

Generate the TTRL and EOPD variants from the DAPO parquet when needed:

```bash
python scripts/data/prepare_dapo_ttrl.py \
  --input datasets/dapo-math-17k.parquet \
  --output datasets/dapo-math-17k-processed.parquet \
  --prompt-format ttrl

python scripts/data/prepare_dapo_ttrl.py \
  --input datasets/dapo-math-17k.parquet \
  --output datasets/dapo-math-17k-raw-question.parquet \
  --prompt-format raw-question
```

All three train datasets are wrapped by the selected student tokenizer using
the explicitly selected chat-template source. Qwen runs use
`apply_chat_template(..., enable_thinking=False)`; models without a thinking
switch record that fact in `run_semantics.json`. EOPD keeps only the original
math question as user content; the tokenizer supplies the chat delimiters and
the assistant prefix.

## 4. Formal training settings

The Qwen EOS/template train entrypoints use:

| Setting | Value |
|---|---:|
| Student | Qwen3-1.7B-Base |
| Teacher | Qwen3-4B |
| GPUs | 4 |
| train batch size | 16 |
| PPO mini-batch size | 16 |
| rollout `n` | 4 |
| response budget | 7168 |
| total training steps | 200 |
| checkpoint frequency | every 20 steps |
| validation during train | disabled (`test_freq=-1`) |
| OPD estimator | sampled token, `log_prob_top_k=0` |
| thinking | disabled |
| W&B project | `OPD Length Inflation` |

Therefore a completed run normally contains checkpoints at steps
`20,40,...,200`:

```text
checkpoint/<experiment_name>/
├── run_semantics.json
├── global_step_20/
├── global_step_40/
└── ...
```

`run_semantics.json` is the train→eval contract. It records the student model,
model/semantic EOS IDs, rollout stop IDs, blocked IDs, EOS mode, thinking flag,
prompt template and train dataset. It describes process-local semantics and
does not modify the downloaded model.

Check one condition without allocating a GPU:

```bash
DRY_RUN=true bash slurm/train/train_ttrl_eos_a_baseline_200step.sl
```

Submit one condition:

```bash
sbatch --export=ALL slurm/train/train_ttrl_eos_a_baseline_200step.sl
```

### 4.1 TTRL EOS conditions

| Condition | Train entrypoint | Rollout stop IDs | Blocked IDs | Loss/sampling change |
|---|---|---|---|---|
| A baseline | `train_ttrl_eos_a_baseline_200step.sl` | model default (`151643`) | none | surface-token OPD |
| B two-stop | `train_ttrl_eos_b_two_stop_200step.sl` | `151643,151645` | none | surface-token OPD |
| C teacher-map | `train_ttrl_eos_c_teacher_map_200step.sl` | `151643` | none | map teacher termination mass to `151643` |
| D semantic-class | `train_ttrl_eos_d_semantic_class_200step.sl` | `151643,151645` | none | aggregate both tokens as one semantic action |
| E canonical | `train_ttrl_eos_e_canonical_200step.sl` | `151643` | `151645` | map mass, mask `151645`, renormalize consistently |

### 4.2 Template baselines

- `train_dapo_baseline_200step.sl`: original DAPO `Answer:` instruction.
- `train_eopd_baseline_200step.sl`: raw question only.

### 4.3 Llama 3.2 and Gemma 3 cross-family runs

The cross-family entrypoints keep the same experimental settings as the Qwen
runs: TTRL data, batch size 16, rollout `n=4`, 7168-token responses,
sampled-token OPD, 200 updates, checkpoints every 20 steps, and W&B project
`OPD Length Inflation`. Only the models and the EOS mode differ. Student and teacher
terminal IDs are discovered at runtime and written to `run_semantics.json`; the
scripts do not hard-code Qwen, Llama, or Gemma token IDs. Section 4.4 explains
which EOS modes each pair supports and how to read a `semantic_class` result.

| Entrypoint | Student | Teacher/template source | EOS mode |
|---|---|---|---|
| `train_ttrl_llama32_3b_base_to_instruct_baseline_200step.sl` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-3B-Instruct` | `baseline` |
| `train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-3B-Instruct` | `semantic_class` |
| `train_ttrl_gemma3_4bpt_to_4bit_baseline_200step.sl` | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | `baseline` |
| `train_ttrl_gemma3_4bpt_to_4bit_semantic_class_200step.sl` | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | `semantic_class` |

Inspect any resolved configuration without allocating a GPU, and submit it
directly as a formal run:

```bash
DRY_RUN=true bash slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
sbatch slurm/train/train_ttrl_llama32_3b_base_to_instruct_semantic_class_200step.sl
```

Each entrypoint's own comment header documents how to run it and exactly what to
change to move it to another cluster.

All four wrappers lower actor and reward dynamic token budgets to 8192, which avoids
the 32768-token actor update that exhausted an earlier four-GPU Llama run. Both wrappers were verified to complete train/update steps on four 96 GiB GPUs
under the full formal load (`batch_size=16`, `rollout.n=4`, response budget
7168), rather than reduced smoke settings.

The Gemma wrapper sets `data.return_multi_modal_inputs=False` because this is a
text-only experiment: Gemma's processor otherwise returns variable-length
`token_type_ids` that cannot be reused after rollout responses are appended. It
also disables actor activation offload because the current VERL offload
scheduler counts Gemma's unused vision FSDP layers and fails during text-only
backward. These two switches are local to the Gemma wrapper; the shared launcher
keeps both defaults enabled for existing Qwen/Llama jobs. With activation
offload disabled, the observed Gemma peak was 74.4 GiB allocated and 85.7 GiB
reserved per 96 GiB GPU. The Gemma header requests the validated 296 GiB host
memory allocation.

Ten checkpoints from a 4B model can be large; set `SAVE_FREQ=200` if only the
final checkpoint is required.

### 4.4 EOS fixes available for Llama and Gemma

The EOS machinery is model-agnostic: `run_semantics.py` discovers the student's
and the teacher's native terminal sets at submit time and stores their ordered
union, and `two_stop`/`semantic_class` operate on a terminal set of any size. No
token id is hard-coded in any wrapper. The discovered sets are:

| Pair | Student native EOS | Teacher native EOS | Semantic set (auto) |
|---|---|---|---|
| Qwen3-1.7B-Base → Qwen3-4B | `151643` | `151645, 151643` | `151643, 151645` |
| Llama-3.2-3B → 3B-Instruct | `128001` | `128001, 128008, 128009` | `128001, 128008, 128009` |
| Gemma-3-4B-PT → 4B-IT | `1, 106` | `1, 106` | `1, 106` |

Which modes each pair supports follows directly from those sets:

| Mode | Qwen | Llama | Gemma |
|---|---|---|---|
| `baseline` | yes | yes | yes |
| `two_stop` | yes | yes | yes (same rollout as baseline) |
| `semantic_class` | yes | yes | yes (loss-only change) |
| `teacher_map` | yes | no | no |
| `canonical` | yes | no | no |

`teacher_map` and `canonical` are pairwise-only by construction. Llama's semantic
set has three tokens and Gemma has two student-native EOS ids, so both modes fail
preflight for those pairs with an explicit error rather than running a silently
wrong experiment.

Set alignment and *usable termination signal* are two different things, and the
two pairs fail in different places:

- **Llama's terminal set is misaligned.** The base student treats only
  `<|end_of_text|>` as terminal while the Instruct teacher ends assistant turns
  with `<|eot_id|>` (128009), a token the baseline student cannot stop on. This
  is the Qwen pathology and `semantic_class` addresses it directly.
- **Gemma's terminal set is aligned, but there is almost nothing to aggregate.**
  `gemma-3-4b-pt` declares the same set as `gemma-3-4b-it`, so `two_stop` is a
  no-op and `semantic_class` changes only the loss. Measured at the positions
  where the student terminated, over the first 5 steps of each baseline run, the
  teacher's mass summed over **all** that pair's terminal tokens — the best case
  any surface-form fix can reach — is:

  | Pair | median | mean | share with mass > 0.5 |
  |---|---:|---:|---:|
  | Qwen3-1.7B-Base → Qwen3-4B | 8.8e-05 | 0.221 | **21.8%** |
  | Gemma-3-4B-pt → 4b-it | 9.0e-05 | 0.013 | **0.7%** |
  | Llama-3.2-3B → 3B-Instruct | 1.3e-02 | 0.137 | 11.0% |

  The shapes differ, not just the levels (share with teacher mass > 0.5, bucketed
  by the student's response length at termination):

  | Pair | 0–64 | 64–256 | 256–1024 | 1024–2048 | 2048+ |
  |---|---:|---:|---:|---:|---:|
  | Qwen | 2% | 11% | 34% | **54%** | 22% |
  | Gemma | 3% | 0% | 1% | **0%** | 0% |
  | Llama | 6% | 22% | 6% | 14% | 6% |

  Qwen's teacher agrees with the student's stop more and more as the answer gets
  longer, which is exactly the regime where aggregating surface forms recovers
  real signal. Gemma is flat near zero at every length: among student stops of
  1024 tokens or more, 38.2% of Qwen's have teacher mass above 0.5 versus **0 of
  Gemma's 79**.

Two explanations for the Gemma result are excluded by measurements the
diagnostic entrypoints in this repository reproduce:

- *The teacher can terminate.* Measured with
  `slurm/diagnostic/measure_teacher_lengths_4x6000pro.sl`,
  `gemma-3-4b-it` stops on **99.95%** of 12800 rollouts of this exact prompt
  stream, always on token 106, at median length 1601 — the most reliable
  terminator of the three teachers, with the tightest length distribution
  (std 601 versus Qwen's 1485).
- *The teacher forward pass is not degenerate.* At step 1 the Gemma run logs
  `actor/entropy` 2.23, `teacher/entropy` 2.12 and `critic/advantages/mean`
  −0.42, against 2.68 / 2.53 / −0.53 for the Qwen baseline. Ordinary-token
  scoring is comparable; the anomaly is specific to EOS.

This was then verified directly rather than left as an inference.
`scripts/diagnostic/verify_teacher_eos_scoring.py` holds the prompt and the
scoring code fixed and varies only who wrote the completion, reading the
teacher's terminal-class probability at the position that predicts the final
token (`slurm/diagnostic/verify_teacher_eos_scoring_1gpu.sl`, 64 prompts, `n=2`):

| Pair | teacher @ its **own** stop | @ random interior (control) | teacher @ **student's** stop |
|---|---:|---:|---:|
| Qwen | 0.993 median, 91.5% > 0.5 | 3.6e-15 | 1.3e-04 median, **21.9%** > 0.5 |
| Gemma | **1.000** median, **100%** > 0.5 | 2.4e-16 | 2.0e-04 median, **2.5%** > 0.5 |
| Llama | 0.817 median, 62.4% > 0.5 | 2.9e-07 | 2.1e-03 median, **6.8%** > 0.5 |

Three things follow. The Gemma teacher is the *most* confident of the three about
its own stops, so the scoring path is sound and the `RETURN_MULTI_MODAL_INPUTS=False`
workaround does not corrupt teacher scoring. The interior control sits near zero
for every pair, so the metric discriminates rather than always reporting high.
And the Qwen arm independently reproduces the in-training diagnostic — 21.9% here
against 21.8% from the training NPZs — so the offline and in-training
measurements agree.

So `gemma-3-4b-it` terminates reliably on its **own** trajectories yet assigns
almost no termination probability on the **student's** trajectories at any
length. The cause is on the student side, and it is about form
rather than capacity or correctness. Counting completions that actually **end**
with a boxed answer — requiring the boxed value to be last, because base students
often restate the prompt's own instruction to box the answer:

| Student | ends with a boxed answer | teacher mass on those |
|---|---:|---|
| Qwen3-1.7B-Base | **16 / 118 (13.6%)** | median 0.999, 14/16 above 0.5 |
| gemma-3-4b-pt | **0 / 112 (0.0%)** | no such completion exists |

Both students are equally wrong on the task (0.8% and 0.0% correct), and the
Gemma student is the *larger* of the two, so neither correctness nor capacity
explains the gap. Qwen3-1.7B-Base already emits instruct-style formatted
solutions that Qwen3-4B recognises as a finished turn; `gemma-3-4b-pt` emits code
fragments, LaTeX noise, multilingual drift and prompt restatements.

This is a distribution-shift failure rather than a surface-token one: a second
mechanism, distinct from the Qwen EOS mismatch, which is what makes this pair
worth running as a control.

The same loss-side fix recovers termination here regardless. Gemma
`semantic_class` drives mean response length from the 7168-token budget down to
roughly 2000 tokens — close to the teacher's own median on these prompts — and
the clip rate from ~100% to near zero (`assets/fix_train_dynamics.png`, panels
a-b), with the student's mass at the last token moving onto `e1` while `e2`
collapses (`assets/terminal_token_grid.png`, panels a-b). Aggregating the
terminal tokens into one action is therefore doing work on this pair even though
its stop set was already aligned and `two_stop` is a no-op, which is what makes
Gemma the clean test of the loss-side change on its own.

### 4.5 Llama 3.2 and Gemma 3 checkpoint eval

There is one checkpoint-eval entrypoint per model pair, and it covers every EOS
mode of that pair. The mode is never selected in the eval file; it is read from
the run being evaluated, so the same file serves the baseline and the
`semantic_class` run:

| Model pair | Eval entrypoint |
|---|---|
| Llama-3.2-3B → 3B-Instruct | `eval_ttrl_llama32_3b_base_to_instruct_ckpts.sl` |
| Gemma-3-4B-PT → 4B-IT | `eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl` |

They run the same evaluator as `eval_checkpoints_4x6000pro.sl` and therefore use
the identical formal defaults listed in section 5.3: checkpoints `20,40,...,200`,
AIME24/AIME25/AMC23 at `@16`, 8192 generated tokens, four independent TP=1 vLLM
engines with batched prompt shards, and one `*.tokens.npz` beside every JSONL.
Nothing model-specific is hard-coded: the student model, rollout stop IDs,
blocked IDs, thinking flag, chat-template source and prompt template all come
from that run's `run_semantics.json`, which is what makes the base student
usable at all — `Llama-3.2-3B` and `gemma-3-4b-pt` carry no chat template of
their own, so eval reuses the instruct model's template exactly as train did.

The only differences from the Qwen eval header are the requested host memory
(160 GiB for Llama, 200 GiB for Gemma) and a required `CHECKPOINT_ROOT`:

```bash
export CHECKPOINT_ROOT="$PWD/checkpoint/opd_ttrl_gemma3_4bpt_to_4bit_baseline_<train-job-id>"
DRY_RUN=true bash slurm/eval/eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl
sbatch --export=ALL slurm/eval/eval_ttrl_gemma3_4bpt_to_4bit_ckpts.sl
```

FSDP checkpoints are merged to HF format under `checkpoint_merged/<run>/` before
generation; ten merged 3–4B models need roughly 60–90 GB of scratch space. Set
`MERGED_ROOT` to relocate them.

### 4.6 K2-Horizon-7B long-horizon runs

The K2 arm is the long-horizon extension: a ~9B student distilled from a ~9B
teacher of the same family, at **400 updates** instead of 200. Read
[`docs/K2_HORIZON.md`](../docs/K2_HORIZON.md) first — K2 needs a one-time setup
step the other families do not (revision-pinned snapshots, derived "Llama-view"
model directories, and two hooks that restore its grouped RMSNorm), because its
official remote code targets transformers >= 5.13 while this environment is
transformers 4.5x / vLLM 0.11.

| Entrypoint | Student revision | Teacher | EOS mode | Updates |
|---|---|---|---|---:|
| `train_ttrl_k2_7b_pretrainfinal_to_main_baseline_400step.sl` | `pretrain_final` | `main` | `baseline` | 400 |
| `train_ttrl_k2_7b_pretrainfinal_to_main_semantic_400step.sl` | `pretrain_final` | `main` | `semantic_class` | 400 |
| `train_ttrl_k2_7b_mid1final_to_main_baseline_200step.sl` | `mid_1_final` | `main` | `baseline` | 200 |
| `train_ttrl_k2_7b_mid1final_to_main_semantic_200step.sl` | `mid_1_final` | `main` | `semantic_class` | 200 |
| `train_ttrl_k2_7b_sft1final_to_main_baseline_200step.sl` | `sft_1_final` | `main` | `baseline` | 200 |
| `train_ttrl_k2_7b_sft1final_to_main_semantic_200step.sl` | `sft_1_final` | `main` | `semantic_class` | 200 |

`pretrain_final -> main` is the headline pair: the student treats only EOD as terminal while
the post-trained teacher ends every turn on EOT — the Qwen pathology at 9B, over a doubled
horizon. `mid_1_final` and `sft_1_final` are stage controls: the terminal-token preference has
already flipped by `mid_1_final`, and they keep the 200-update horizon.

Everything else (TTRL data, batch size 16, rollout `n=4`, response budget 7168,
lr 1e-6, sampled-token OPD, save every 20 steps) matches the other families. The
K2-specific settings are all in the wrappers: `REWARD_MICRO_BATCH_SIZE=4` and a
larger `--mem` for the 250k-vocab 9B pair, `OPD_K2_GROUPED_RMSNORM=1`, and
`PYTHONPATH=scripts/k2`.

```bash
export K2_MODELS_ROOT=/large/filesystem/k2-models   # required; ~18 GB per revision plus its view
DRY_RUN=true bash slurm/train/train_ttrl_k2_7b_pretrainfinal_to_main_semantic_400step.sl
sbatch --export=ALL slurm/train/train_ttrl_k2_7b_pretrainfinal_to_main_semantic_400step.sl
```

`TOTAL_TRAINING_STEPS=<n>` overrides the horizon on any wrapper. Twenty checkpoints of a 9B
model are large; set `SAVE_FREQ=400` if only the final checkpoint is needed.

One eval entrypoint covers every K2 run and every EOS mode — the student, template,
stop/blocked ids and thinking flag all come from that run's `run_semantics.json`. It defaults
to `20,40,...,400`; the 200-update stage controls need `FINAL_STEP=200`:

```bash
export CHECKPOINT_ROOT="$PWD/checkpoint/opd_ttrl_k2_7b_pretrainfinal_to_main_semantic_class_<job-id>"
sbatch --export=ALL slurm/eval/eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl

CHECKPOINT_ROOT="$PWD/checkpoint/opd_ttrl_k2_7b_mid1final_to_main_baseline_<job-id>" FINAL_STEP=200 \
  sbatch --export=ALL slurm/eval/eval_ttrl_k2_7b_pretrainfinal_to_main_ckpts.sl
```

K2 eval forces `EVAL_MAX_TOKENS=7168` and `EVAL_MAX_MODEL_LEN=8192` (the
student's native context is 8192, against 8192/12288 for Llama and Gemma);
override with `K2_EVAL_MAX_TOKENS` / `K2_EVAL_MAX_MODEL_LEN` only deliberately.
Measuring a truncation rate requires `max_tokens < max_model_len`, otherwise the
`len >= max_tokens` rule can never fire and silently reports zero.

## 5. Train and eval coupling

The train and eval implementations are deliberately decoupled: each can be
run, retried, or moved to a different GPU type independently. They are coupled
only through the immutable `checkpoint/<run>/run_semantics.json` manifest.

### 5.1 Automatic TTRL EOS comparison

The submission helper launches five train jobs and five matching eval jobs.
Each eval uses Slurm `afterok:<train_job_id>`, so it starts only after its train
job completes successfully and automatically receives the exact checkpoint
root for that condition:

```bash
DRY_RUN=true bash slurm/submit_ttrl_eos_sweep.sh
bash slurm/submit_ttrl_eos_sweep.sh
```

Set a stable sweep label when desired:

```bash
SWEEP_TAG=eos-reproduce-v1 bash slurm/submit_ttrl_eos_sweep.sh
```

The helper prints all ten job IDs, dependency edges, experiment names and
checkpoint paths. It does not run train or eval inside the submit/login process.

### 5.2 Llama and Gemma runs in one command

`submit_multimodel_runs.sh` does the same for the cross-family runs: for each
selected (model, EOS mode) pair it submits the train job and chains that model's
eval entrypoint with `afterok:<train-job-id>`, so the eval starts only after its
own train job succeeds and already knows the right checkpoint root.

The default selection is `MODELS=llama,gemma` and `EOS_MODES=baseline,semantic_class`,
i.e. four train jobs and four evals. Always inspect the plan first:

```bash
DRY_RUN=true bash slurm/submit_multimodel_runs.sh
bash slurm/submit_multimodel_runs.sh
```

Restrict the selection or set a stable run label:

```bash
MODELS=gemma bash slurm/submit_multimodel_runs.sh
EOS_MODES=semantic_class bash slurm/submit_multimodel_runs.sh
RUN_TAG=multimodel-v1 bash slurm/submit_multimodel_runs.sh
```

Section 4.4 explains what the Gemma `semantic_class` run does and does not change.

Authenticate with `hf auth login` and `wandb login` first; both model families
are gated and the default W&B mode is online.

### 5.3 Manual checkpoint eval

For an independently submitted train run:

```bash
export CHECKPOINT_ROOT="$PWD/checkpoint/<experiment_name>"
sbatch --export=ALL slurm/eval/eval_checkpoints_4x6000pro.sl
```

The formal defaults are:

- checkpoints `20,40,...,200`;
- AIME24, AIME25 and AMC23;
- `@16` sampling per problem;
- non-thinking inherited from train;
- 8192 generated tokens;
- four independent TP=1 vLLM engines with batched prompt shards;
- one `*.tokens.npz` file next to every generated JSONL file.

Do not set EOS manually for checkpoint eval. It reads all of the following from
`run_semantics.json`:

- `model_path` (unless `BASE_MODEL` is explicitly supplied for relocation);
- `rollout_stop_token_ids`;
- `blocked_token_ids`;
- `enable_thinking`;
- `prompt_template`;
- `eos_mode` for logging/auditing.

The checkpoint evaluator fails if the manifest is missing. This prevents a run
trained with one EOS policy from being silently evaluated with another.
Each eval output root also contains `eval_run_config.json`, which fingerprints
the resolved template, EOS stop/block IDs, thinking flag, model/data identity,
sampling settings and engine layout. Existing generations are reused only when
that config matches and the JSONL has a valid matching token NPZ. Use a distinct
`OUTPUT_ROOT` for a cross-template ablation; mismatched output reuse fails fast.

Inspect the resolved settings without GPUs or checkpoint shards:

```bash
CHECKPOINT_ROOT="$PWD/checkpoint/<experiment_name>" DRY_RUN=true \
  bash slurm/eval/eval_checkpoints_4x6000pro.sl
```

## 6. Direct Hugging Face model eval

Use `eval_model_4x6000pro.sl` for a base model or an already merged HF model.
There is no train manifest in this case, so generation semantics must be
selected explicitly with `EOS_MODE` (or explicit stop/block IDs):

```bash
export MODEL_PATH=Qwen/Qwen3-1.7B-Base
export EOS_MODE=baseline
export EVAL_TEMPLATE=ttrl
sbatch --export=ALL slurm/eval/eval_model_4x6000pro.sl
```

Important variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | required | HF ID or local merged-model directory |
| `EOS_MODE` | `baseline` | generation EOS mode |
| `SEMANTIC_EOS_TOKEN_IDS` | `151643,151645` | semantic `e1,e2` |
| `STOP_TOKEN_IDS` | derived | explicit override; empty means tokenizer default |
| `BLOCKED_TOKEN_IDS` | derived | IDs masked during sampling |
| `ENABLE_THINKING` | `false` | Qwen thinking template flag |
| `EVAL_TEMPLATE` | `ttrl` | `ttrl`, `dapo`, or `eopd` |
| `EVAL_TASKS` | `AIME24,AIME25,AMC23` | comma-separated tasks |
| `EVAL_N` | `16` | rollouts per problem (`@N`) |
| `EVAL_MAX_TOKENS` | `8192` | response budget |
| `OUTPUT_ROOT` | generated | JSONL, token IDs and summaries |

Generation mapping:

| `EOS_MODE` | Stop IDs | Blocked IDs |
|---|---|---|
| `baseline` | tokenizer/model default | none |
| `two_stop` | `151643,151645` | none |
| `teacher_map` | `151643` | none |
| `semantic_class` | `151643,151645` | none |
| `canonical` | `151643` | `151645` |

`teacher_map` and `semantic_class` contain teacher/loss behavior during train;
eval reproduces only their student generation stop/block semantics.

Dry-run example:

```bash
MODEL_PATH=Qwen/Qwen3-1.7B-Base EOS_MODE=canonical DRY_RUN=true \
  bash slurm/eval/eval_model_4x6000pro.sl
```

## 7. Checkpoint watcher

The L40S watcher polls a live run and evaluates every newly complete checkpoint:

```bash
sbatch slurm/eval/watch_checkpoints_4xl40s.sl \
  --checkpoint-root "$PWD/checkpoint/<experiment_name>" \
  --final-step 200 \
  --poll-seconds 30
```

The watcher waits for `data.pt`, HF metadata, FSDP config and every model shard.
By default it inherits EOS, thinking and the exact train prompt template from
`run_semantics.json`. `--eval-templates ttrl,dapo,eopd` is an explicit override
for a deliberate cross-template evaluation, not the default.

## 8. Two-stage EOS/template experiment

Recommended experiment order:

1. Run `submit_ttrl_eos_sweep.sh`. All five conditions use identical TTRL data,
   model, batch size, budget and eval template; only EOS behavior changes.
2. Compare AIME24, AIME25 and AMC23 scores, their unweighted macro average,
   sequence length statistics and EOS diagnostics. Select one EOS condition.
3. Freeze that selected `EOS_MODE` in three independent formal train wrappers:
   TTRL, DAPO and EOPD. Do not compare templates while also changing EOS.
4. Eval each run using the template recorded in its manifest: TTRL→TTRL,
   DAPO→DAPO, EOPD→EOPD. Use an explicit override only for a separately named
   cross-template ablation.

The shipped DAPO and EOPD wrappers deliberately use `EOS_MODE=baseline` rather
than pre-committing to a winner: which EOS condition step 2 selects depends on
the sweep. To run step 3 with a different mode, copy the small wrapper file,
change only `EOS_MODE`, and give the copy a filename that records the mode — the
shared launcher keeps every other setting identical, which is what makes the
template comparison a single-variable one.

## 9. Base-model GPU smoke

```bash
sbatch --export=ALL slurm/smoke/smoke_eval_base_4x6000pro.sl
```

This is Qwen3-1.7B-Base, baseline EOS, TTRL non-thinking, AMC23 `@1`, and a
256-token budget. It checks imports, CUDA/vLLM, multiprocessing, batched
generation, token-ID persistence and grading. It is not a performance result.

## 10. Eval outputs and reporting

```text
eval_outputs/<run>/
├── eval_run_config.json
├── step_XXXX/
│   ├── aime24_*.jsonl
│   ├── aime24_*.tokens.npz
│   ├── aime25_*.jsonl
│   ├── aime25_*.tokens.npz
│   ├── amc23_*.jsonl
│   └── amc23_*.tokens.npz
├── grading_summary.json
└── grading_summary.csv
```

Each NPZ contains flat `token_ids`, `offsets`, and the paired JSONL SHA-256,
preserving every generated trajectory without changing the JSONL schema and
preventing a stale/interrupted JSONL/NPZ pair from being reused. Report the overall OPD score as
the unweighted arithmetic mean of the AIME24, AIME25 and AMC23 `mean_score`
values, not weighted by their different problem counts.

## 11. Troubleshooting

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

- `Missing train/eval semantics manifest`: `CHECKPOINT_ROOT` is not a train-run
  root, or the run predates the manifest contract.
- `Missing checkpoint shard`: the save is incomplete; wait or use the watcher.
- `python not found`: activate the environment before `sbatch` or set
  `CONDA_ENV`.
- `nvcc not found`: load a CUDA toolkit or set `CUDA_MODULE`.
- vLLM OOM: lower `EVAL_MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION`; for these
  small models keep TP=1 and shard prompts across independent engines.
- W&B cannot connect: set `WANDB_MODE=offline` and sync later.
