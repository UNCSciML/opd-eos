"""K2-Horizon dense checkpoints for vLLM: Llama architecture with grouped RMSNorm.

Mirrors K2HorizonRMSNorm from the official modeling_k2_horizon.py: the hidden
dimension is split into `layernorm_num_groups` contiguous groups and each group
is normalised by its own RMS; weight is applied in fp32 before casting back.
The fused residual-add convention of vLLM's RMSNorm (forward(x, residual) ->
(out, residual)) is preserved so the Llama decoder layer code is unchanged.
"""
from typing import Optional

import torch
from torch import nn

from vllm.model_executor.models.llama import LlamaForCausalLM


class GroupedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, n_groups: int, eps: float = 1e-6):
        super().__init__()
        assert hidden_size % n_groups == 0
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x32: torch.Tensor) -> torch.Tensor:
        h = x32.reshape(*x32.shape[:-1], self.n_groups, -1)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return h.reshape(*h.shape[:-2], -1)

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        orig_dtype = x.dtype
        if residual is not None:
            # official K2/HF decoder layers add the residual in the working dtype
            x = x + residual
            residual = x
        x32 = x.to(torch.float32)
        out = (self.weight.to(torch.float32) * self._norm(x32)).to(orig_dtype)
        return out if residual is None else (out, residual)

    def extra_repr(self):
        return f"hidden_size={self.hidden_size}, n_groups={self.n_groups}, eps={self.variance_epsilon}"


def _swap_norms(model: nn.Module, hidden_size: int, n_groups: int, eps: float):
    for layer in model.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            if hasattr(layer, name) and not isinstance(getattr(layer, name), GroupedRMSNorm):
                setattr(layer, name, GroupedRMSNorm(hidden_size, n_groups, eps))
    if hasattr(model.model, "norm") and isinstance(model.model.norm, nn.Module) and hasattr(model.model.norm, "weight"):
        model.model.norm = GroupedRMSNorm(hidden_size, n_groups, eps)


class K2HorizonForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        cfg = vllm_config.model_config.hf_config
        n_groups = int(getattr(cfg, "layernorm_num_groups", 1))
        if n_groups > 1:
            _swap_norms(self, cfg.hidden_size, n_groups, cfg.rms_norm_eps)
