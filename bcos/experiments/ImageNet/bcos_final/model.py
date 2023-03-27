from functools import partial

import bcos.models.convnext as convnext
import bcos.models.densenet as densenet
import bcos.models.resnet as resnet
import bcos.models.vgg as vgg
from bcos.modules.bcosconv2d import BcosConv2d

__all__ = ["get_model"]


def get_arch_builder(arch_name: str):
    arch_builder = None

    if arch_name.startswith("resne"):
        arch_builder = getattr(resnet, arch_name)
    elif arch_name.startswith("densenet"):
        arch_builder = getattr(densenet, arch_name)
    elif arch_name.startswith("vgg"):
        arch_builder = getattr(vgg, arch_name)
    elif arch_name.startswith("convnext"):
        arch_builder = getattr(convnext, arch_name)

    assert arch_builder is not None
    return arch_builder


def get_model(model_config):
    assert model_config.get("is_bcos", False), "Should be true!"
    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]
    bcos_args = model_config["bcos_args"]

    # specify conv layer
    if "conv_layer" not in args:  # for compatibility with hubconf overrides
        args["conv_layer"] = partial(BcosConv2d, **bcos_args)

    assert "norm_layer" in args, "norm_layer is required!"

    # create model
    model = get_arch_builder(arch_name)(**args)

    return model
