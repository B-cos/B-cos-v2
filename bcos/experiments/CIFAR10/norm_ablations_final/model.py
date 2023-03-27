from functools import partial

import bcos.models.resnet
from bcos.modules.bcosconv2d import BcosConv2d, BcosConv2dWithScale

__all__ = ["get_model"]


def get_model(model_config):
    assert model_config.get("is_bcos", False), "Should be true!"
    # extract args
    arch_name = model_config["name"]
    args = model_config["args"]
    bcos_args = model_config["bcos_args"]

    # specify conv layer
    # unlike norm_layer these stay the same class
    if model_config.get("use_bcos_scale", False):
        args["conv_layer"] = partial(BcosConv2dWithScale, **bcos_args)
    else:
        args["conv_layer"] = partial(BcosConv2d, **bcos_args)

    # create model
    model = getattr(bcos.models.resnet, f"cifar10_{arch_name}")(**args)

    return model
