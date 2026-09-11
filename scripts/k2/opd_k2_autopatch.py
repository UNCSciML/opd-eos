"""Import-time hook: apply the K2 grouped-RMSNorm patch when OPD_K2_GROUPED_RMSNORM=1.

Lazy: nothing heavy is imported at interpreter start. A meta-path finder waits for
`transformers.models.llama.modeling_llama` to be imported and applies the patch right
after that module executes. (Eagerly importing transformers here made every Python
process — including Ray's raylet agents — take seconds to start, which broke Ray head
startup on busy nodes.)
"""
import os, sys, importlib.abc, importlib.util

_TARGET = "transformers.models.llama.modeling_llama"


class _K2LazyPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        real_loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return real_loader.create_module(spec)

            def exec_module(self, module):
                real_loader.exec_module(module)
                try:
                    import opd_k2_patch
                    opd_k2_patch.apply()
                except Exception as e:  # never break the host process
                    print(f"[opd_k2_autopatch] failed: {e!r}", file=sys.stderr)

        spec.loader = _Loader()
        return spec


if os.environ.get("OPD_K2_GROUPED_RMSNORM") == "1" and not any(isinstance(f, _K2LazyPatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _K2LazyPatchFinder())
