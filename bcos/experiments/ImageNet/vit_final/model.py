from functools import partial

from torch import nn

import bcos.models.vit as vit
from bcos.modules.bcosconv2d import BcosConv2d
from bcos.modules.bcoslinear import BcosLinear
from bcos.modules.common import BcosSequential
from bcos.modules.logitlayer import LogitLayer

__all__ = ["get_model"]


def get_arch_builder(arch_name: str):
    arch_builder = getattr(vit, arch_name)

    assert arch_builder is not None
    return arch_builder


def get_model(model_config):
    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]

    is_bcos = model_config["is_bcos"]

    # specify linear layer and conv2d_layer
    if (
        "linear_layer" not in args or "conv2d_layer" not in args
    ):  # for compatibility with hubconf overrides
        if is_bcos:
            bcos_args = model_config.get("bcos_args", dict())
            args["linear_layer"] = partial(BcosLinear, **bcos_args)
            args["conv2d_layer"] = partial(BcosConv2d, **bcos_args)
        else:
            args["linear_layer"] = nn.Linear
            args["conv2d_layer"] = nn.Conv2d

    assert "norm_layer" in args, "norm_layer is required!"

    # create model
    model = get_arch_builder(arch_name)(**args)

    if is_bcos:
        logit_bias = model_config["logit_bias"]
        model = BcosSequential(model, LogitLayer(logit_bias=logit_bias))

    return model
