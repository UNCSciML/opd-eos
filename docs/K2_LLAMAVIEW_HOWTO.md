# K2-Horizon-7B 的 "Llama-view"：它是什么、怎么做、怎么用 vLLM 跑 teacher

> 目标：在 **vLLM 0.11 / transformers 4.56** 这套 OPD 环境里跑 `IFM/K2-Horizon-7B`（例如用 `main` 当 teacher，测训练 prompt 上的长度 / clip rate）。
> 官方 K2 只提供 transformers ≥ 5.13 的 remote code（`K2HorizonForCausalLM`），vLLM 里没有这个架构，所以我们自己做了一个 "Llama-view"。**它是我们内部的构造，不是 IFM 官方的。**

## 1. 原理：K2-7B dense = Llama + 分组 RMSNorm

对官方 config / modeling 代码逐项核对过，K2-Horizon-7B（dense 版）和 Llama 结构完全一致：0 个 expert、全 MLP 层、无 q/k-norm、无 sliding window、silu、head_dim 128、标准 RoPE、GQA（32 头 / 8 KV 头）、36 层、hidden 4096、vocab 250 624。
**唯一区别**是 RMSNorm：K2 把 hidden 维切成 `layernorm_num_groups = 4` 个连续分组，每组各自算 RMS 再归一化（`K2HorizonRMSNorm`）；Llama 是整条向量一起归一化。

因此做法是：
1. **权重原封不动**（safetensors 直接 symlink，参数名和 Llama 完全相同）；
2. 写一份 **Llama 格式的 config**（`model_type: "llama"`，保留 `layernorm_num_groups: 4`，`architectures: ["K2HorizonForCausalLM"]`）；
3. 用两个很小的 hook 把分组 RMSNorm 补回来：
   * **vLLM**：一个 out-of-tree 模型插件（`scripts/k2/vllm_plugin`，entry point `vllm.general_plugins`），把 `K2HorizonForCausalLM` 注册为 "vLLM 的 `LlamaForCausalLM` + `GroupedRMSNorm`"；vLLM 每个 worker 进程都会自动加载它。
   * **HF/transformers 4.x（训练侧 actor / reward）**：`scripts/k2/opd_k2_patch.py`，只在 `OPD_K2_GROUPED_RMSNORM=1` 时把 `LlamaRMSNorm.forward` 换成分组版本，通过一个懒加载的 `.pth` hook 装进 site-packages。

**正确性验证**（用 `scripts/k2/k2_equivalence.py` + `k2_equivalence_compare.py` 复现）：官方 remote code（transformers 5.15 参考环境）vs 我们的 patched HF：FP32 和 BF16 都 **bit-exact**（max|Δlogit| = 0，全部隐层 = 0，`main` 和 `pretrain_final` 都测了）；patched vLLM vs 官方：在 vLLM 后端数值噪声以内（末位置 max|Δlogp| 0.44，噪声底 0.52；top-1 16/16 一致）。只用 Llama 的 RMSNorm 不加分组时会明显偏离（对照实验 max|Δlogit| 0.75），所以插件不能省。

## 2. 需要的文件

| 文件 | 作用 |
|---|---|
| `scripts/diagnostics/make_k2_llama_view.py` | 从下载的官方 snapshot 生成 Llama-view 目录 |
| `scripts/k2/vllm_plugin/` | vLLM 插件（`pip install -e`） |
| `scripts/k2/opd_k2_patch.py`, `scripts/k2/opd_k2_autopatch.py` | HF 分组 RMSNorm patch + 懒加载 hook |
| `scripts/k2/install_k2_hooks.sh` | 一键安装上面两个 hook 并自检 |
| `scripts/k2/k2_teacher_length_vllm.py` | 用 vLLM 测某个 prompt 集合上的长度 / clip rate / 终止 token 分布 |
| `scripts/diagnostics/check_k2_eos.py` | 打印 EOS / 模板元数据 |
| `scripts/k2/k2_equivalence.py`, `k2_equivalence_compare.py` | 复现等价性测试（官方侧需要 transformers ≥ 5.13 的独立 venv） |

## 3. 一次性准备

```bash
# 环境：vllm 0.11.0, transformers 4.56.1, torch 2.8.0（我们实测的组合）
export K2=/path/with/lots/of/space          # 每个 revision 约 17–18 GB

# (1) 下载官方 snapshot（按 revision 固定；main 的 sha 是 586b03f0…）
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for rev in ["main"]:                          # 需要 student 再加 "pretrain_final" 等
    snapshot_download("IFM/K2-Horizon-7B", revision=rev, local_dir=f"{os.environ['K2']}/K2-Horizon-7B-{rev}")
PY

# (2) 生成 Llama-view（权重 symlink，不复制；tokenizer 用 main 的）
python scripts/diagnostics/make_k2_llama_view.py \
    --src $K2/K2-Horizon-7B-main --tokenizer-src $K2/K2-Horizon-7B-main --out $K2/K2-Horizon-7B-main-llamaview

# (3) 在 vLLM 所在的 python 环境里装 hook（vLLM 插件 + HF patch），自带自检
bash scripts/k2/install_k2_hooks.sh
```

`make_k2_llama_view.py` 还会：断言该 snapshot 的 config 确实是 "Llama + 分组 norm"（有别的差异会拒绝生成）；把官方聊天模板复制为 `chat_template.upstream.jinja`，并生成一份 **派生模板** `chat_template.jinja`：多了一个 `enable_thinking=False` 开关（见 §5）。不传这个参数时行为和官方模板完全相同。

## 4. 用 vLLM 跑 teacher（main），测训练 prompt 上的长度 / clip rate

```bash
python scripts/k2/k2_teacher_length_vllm.py \
    --model $K2/K2-Horizon-7B-main-llamaview \
    --prompts train_prompts.jsonl --field prompt \
    --n 4 --max-tokens 7168 --max-model-len 8192 --temperature 1.0 --top-p 1.0 \
    --out teacher_main_len.json          # 加 --thinking 用官方思考模板；加 --raw 不套模板
```
它做的事就是下面这几行，可以直接抄进自己的代码：
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{K2}/K2-Horizon-7B-main-llamaview")
prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)   # 训练里用的非思考形式
llm = LLM(model=f"{K2}/K2-Horizon-7B-main-llamaview", dtype="bfloat16", max_model_len=8192,
          trust_remote_code=False)                     # 不需要 remote code；插件通过 entry point 自动加载
sp  = SamplingParams(n=4, max_tokens=7168, temperature=1.0, top_p=1.0, stop_token_ids=[1, 250019])
outs = llm.generate([prompt], sp)
clip = sum(c.finish_reason == "length" for o in outs for c in o.outputs)
```
关键参数说明：
* **stop ids = [1, 250019]**：`1 = <|ifm|endoftext|>`（EOD），`250019 = <|ifm|im_end|>`（EOT）。所有 revision 的 `generation_config.eos_token_id` 都是这两个，所以这就是"官方停止集"；vLLM 也会自动读 generation_config，但显式写上更稳。
* **max_model_len**：`main` 的 `max_position_embeddings` 是 524 288（rope_theta 1e7），想设多大都行；我们训练用 prompt 1024 + response 7168 = 8192，是为了迁就 student `pretrain_final`（它的 max_pos 只有 8192，rope_theta 5e5）。只测 teacher 的话可以放宽。
* **clip rate** 用 `finish_reason == "length"` 判断，不要用 `len(token_ids) >= max_tokens`（当 prompt + max_tokens 超过 max_model_len 时后者永远不成立，我们在 eval 里踩过这个坑）。
* 参考数：`main` 在非思考模板、greedy、≤ 4096 token 下 70 % 能终止（全部以 EOT 结束），AMC23 87.5 % / AIME24 43.3 %，终止位置 p(EOT) 0.98（用 `scripts/k2/k2_teacher_sanity.py` 复现）。训练同分布采样（T = 1.0, 7168）下的长度就是这个脚本要测的。

## 5. 模板 / tokenizer 的几个坑

* K2 官方模板是 **默认思考**（没有 `enable_thinking` 开关，只有 reasoning_effort → `<ifm|think>` / `think_fast` / `think_faster`）。派生模板加了 `enable_thinking=False`：generation prompt 变成模板自己的"空思考"形式
  `<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>`（ids … 250029, 200, 250030），这就是 OPD 训练里 student / teacher 看到的 prompt 结尾。不传参数则和官方一致（`…assistant\n<ifm|think>\n`）。
* tokenizer 的 vocab / merges / id 在所有 revision 完全相同，但 24 个特殊 token 的**字面**在 main 里改了名（`<|im_end|>` → `<|ifm|im_end|>` 等），所以统一用 **main 的 tokenizer**（`--tokenizer-src`）；普通文本 700/700 条 id 完全一致。
* `bos_token_id = 0`（`<|ifm|begin_of_text|>`）；聊天模板自己不加 BOS，是否加由你决定，保持和训练一致即可。
* 模型 9.0 B 实参（大 vocab），bf16 单卡 ≥ 24 GB 可跑；我们用 4 卡 tensor_parallel=1 各跑一份做数据并行。

## 6. 不用 vLLM 的替代

transformers ≥ 5.13 + `trust_remote_code=True` 直接加载官方 repo 即可（慢，适合小规模验证）。训练出来的 student checkpoint 可以用 `scripts/k2/export_k2_official_format.py` 转回这种官方格式，之后不需要任何 hook。
