"""vLLM general plugin: registers K2HorizonForCausalLM (Llama + grouped RMSNorm).

Loaded automatically by vLLM in every engine/worker process via the
`vllm.general_plugins` entry point.  The model class is imported lazily.
"""


def register():
    from vllm import ModelRegistry
    if "K2HorizonForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("K2HorizonForCausalLM", "opd_k2_vllm_plugin.model:K2HorizonForCausalLM")
