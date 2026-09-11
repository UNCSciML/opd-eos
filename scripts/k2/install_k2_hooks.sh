#!/usr/bin/env bash
# One-time environment setup for K2-Horizon runs: vLLM plugin + env-gated HF grouped-RMSNorm hook.
set -euo pipefail
PR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
pip install -e "$PR/scripts/k2/vllm_plugin"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
printf '%s\nimport opd_k2_autopatch\n' "$PR/scripts/k2" > "$SP/opd_k2.pth"
python -c "from vllm.plugins import load_general_plugins; load_general_plugins(); from vllm import ModelRegistry; assert 'K2HorizonForCausalLM' in ModelRegistry.get_supported_archs(); print('vLLM plugin OK')"
OPD_K2_GROUPED_RMSNORM=1 python -c "import transformers.models.llama.modeling_llama as m; assert m.LlamaRMSNorm.forward.__name__=='grouped_rmsnorm_forward'; print('HF hook OK')"
