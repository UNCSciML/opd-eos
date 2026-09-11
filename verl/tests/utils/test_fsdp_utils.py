import torch.nn as nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class PresentDecoderLayer(nn.Module):
    pass


class ModelWithOptionalNoSplitModule(nn.Module):
    _no_split_modules = ["PresentDecoderLayer", "OptionalPoolingLayer"]

    def __init__(self):
        super().__init__()
        self.decoder = PresentDecoderLayer()


def test_fsdp_wrap_policy_ignores_absent_optional_no_split_module():
    policy = get_fsdp_wrap_policy(ModelWithOptionalNoSplitModule())

    assert callable(policy)
