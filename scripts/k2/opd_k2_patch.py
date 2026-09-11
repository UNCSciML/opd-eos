"""Grouped RMSNorm patch for the K2-Horizon Llama-view (HF/transformers side).

K2-Horizon-7B is architecturally Llama except that its RMSNorm normalises over
`layernorm_num_groups` contiguous groups of the hidden dimension (variance per
group) — see K2HorizonRMSNorm in the official modeling_k2_horizon.py.  When a
LlamaConfig carries `layernorm_num_groups > 1` (written by
scripts/diagnostics/make_k2_llama_view.py), this patch makes every LlamaRMSNorm
in the model apply the grouped variant.  Models whose config lacks the attribute
(plain Llama, Qwen, Gemma, ...) are untouched.

Activate with `apply()` (idempotent) — done automatically by sitecustomize-style
hook `opd_k2_autopatch` when OPD_K2_GROUPED_RMSNORM=1.
"""
import torch

_APPLIED = False


def grouped_rmsnorm_forward(self, hidden_states):
    n_groups = getattr(self, "n_groups", 1)
    if n_groups <= 1:
        return _orig_forward(self, hidden_states)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], n_groups, -1)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
    return self.weight * hidden_states.to(input_dtype)


def _tag_norms(model, n_groups):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    for m in model.modules():
        if isinstance(m, LlamaRMSNorm):
            m.n_groups = int(n_groups)


def apply():
    global _APPLIED, _orig_forward
    if _APPLIED:
        return
    from transformers.models.llama import modeling_llama as ml
    _orig_forward = ml.LlamaRMSNorm.forward
    ml.LlamaRMSNorm.forward = grouped_rmsnorm_forward

    _orig_model_init = ml.LlamaModel.__init__

    def patched_model_init(self, config):
        _orig_model_init(self, config)
        _tag_norms(self, getattr(config, "layernorm_num_groups", 1))

    ml.LlamaModel.__init__ = patched_model_init
    _APPLIED = True


# NOTE on the official semantics (K2HorizonRMSNorm): weight is applied AFTER
# casting back to the input dtype: `self.weight * hidden_states.to(input_dtype)`
# where the cast happens on the normalised fp32 tensor.  The official code does
# `hidden_states = self.weight * hidden_states; return hidden_states.to(input_dtype)`
# i.e. weight multiplication in fp32 then cast.  We follow the official order:
def grouped_rmsnorm_forward(self, hidden_states):  # noqa: F811  (final definition wins)
    n_groups = getattr(self, "n_groups", 1)
    if n_groups <= 1:
        return _orig_forward(self, hidden_states)
    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    h = h.reshape(*h.shape[:-1], n_groups, -1)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + self.variance_epsilon)
    h = h.reshape(*h.shape[:-2], -1)
    h = self.weight * h
    return h.to(input_dtype)
