"""
This module contains entrypoints to loading pretrained models.

Notes
-----
Doing it this way allows:
#. a better overview of the available weights
#. less coupling of model definition code and our config system
#. easy integration with torch.hub

There is obviously a tradeoff here, as this requires slightly more code
but given the advantages I think it's worth it.
"""
from typing import Any, Callable, Dict, List

import torch

from bcos.experiments.utils import Experiment

__all__ = ["list_available"]  # entrypoints filled by `register` decorator


_entrypoint_registry = {}


def register(
    entrypoint_fn: Callable[..., torch.nn.Module]
) -> Callable[..., torch.nn.Module]:
    """Decorator to register a function as an entrypoint."""
    _entrypoint_registry[entrypoint_fn.__name__] = entrypoint_fn
    __all__.append(entrypoint_fn.__name__)
    return entrypoint_fn


def list_available() -> List[str]:
    """List all available pretrained models."""
    return [k for k in _entrypoint_registry.keys()]


BASE = "https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights"


# map (base -> (model_name -> url))
URLS: Dict[str, Dict[str, str]] = {
    "bcos_final": {
        # resnets and resnext
        "resnet_18": f"{BASE}/resnet_18-68b4160fff.pth",
        "resnet_34": f"{BASE}/resnet_34-a63425a03e.pth",
        "resnet_50": f"{BASE}/resnet_50-ead259efe4.pth",
        "resnet_101": f"{BASE}/resnet_101-84c3658278.pth",
        "resnet_152": f"{BASE}/resnet_152-42051a77c1.pth",
        "resnext_50_32x4d": f"{BASE}/resnext_50_32x4d-57af241ab9.pth",
        # densenets
        "densenet_121": f"{BASE}/densenet_121-b8daf96afb.pth",
        "densenet_161": f"{BASE}/densenet_161-9e9ea51353.pth",
        "densenet_169": f"{BASE}/densenet_169-7037ee0604.pth",
        "densenet_201": f"{BASE}/densenet_201-00ac87066f.pth",
        # other
        "vgg_11_bnu": f"{BASE}/vgg_11_bnu-34036029f0.pth",
    },
    "bcos_final_long": {
        "convnext_tiny_pn": f"{BASE}/convnext_tiny_pn-539b1bfb37.pth",
        "convnext_base_pn": f"{BASE}/convnext_base_pn-b0495852c6.pth",
        "convnext_tiny_bnu": f"{BASE}/convnext_tiny_bnu-dbd7f5ef9d.pth",
        "convnext_base_bnu": f"{BASE}/convnext_base_bnu-7c32a704b3.pth",
        "densenet_121": f"{BASE}/densenet_121_long-5175461597.pth",
        "resnet_50": f"{BASE}/resnet_50_long-ef38a88533.pth",
        "resnet_152": f"{BASE}/resnet_152_long-0b4b434939.pth",
    },
    "vit_final": {
        "bcos_simple_vit_ti_patch16_224": f"{BASE}/bcos_simple_vit_ti_patch16_224-4b0824b1c1.pth",
        "bcos_simple_vit_s_patch16_224": f"{BASE}/bcos_simple_vit_s_patch16_224-75e99d1f73.pth",
        "bcos_simple_vit_b_patch16_224": f"{BASE}/bcos_simple_vit_b_patch16_224-1fc4750806.pth",
        "bcos_simple_vit_l_patch16_224": f"{BASE}/bcos_simple_vit_l_patch16_224-9613b2ad0a.pth",
        "bcos_vitc_ti_patch1_14": f"{BASE}/bcos_vitc_ti_patch1_14-ddd6193a77.pth",
        "bcos_vitc_s_patch1_14": f"{BASE}/bcos_vitc_s_patch1_14-cf55c88f0c.pth",
        "bcos_vitc_b_patch1_14": f"{BASE}/bcos_vitc_b_patch1_14-a13c46397b.pth",
        "bcos_vitc_l_patch1_14": f"{BASE}/bcos_vitc_l_patch1_14-8739e18b8d.pth",
        # standard! ie non-B-cos
        "simple_vit_ti_patch16_224": f"{BASE}/standard_simple_vit_ti_patch16_224-2ae8c65a39.pth",
        "simple_vit_s_patch16_224": f"{BASE}/standard_simple_vit_s_patch16_224-f2934fcdcf.pth",
        "simple_vit_b_patch16_224": f"{BASE}/standard_simple_vit_b_patch16_224-87074200ed.pth",
        "simple_vit_l_patch16_224": f"{BASE}/standard_simple_vit_l_patch16_224-62dc536e03.pth",
        "vitc_ti_patch1_14": f"{BASE}/standard_vitc_ti_patch1_14-a5d6bded37.pth",
        "vitc_s_patch1_14": f"{BASE}/standard_vitc_s_patch1_14-34ecd7288e.pth",
        "vitc_b_patch1_14": f"{BASE}/standard_vitc_b_patch1_14-4d374b0220.pth",
        "vitc_l_patch1_14": f"{BASE}/standard_vitc_l_patch1_14-560e48f246.pth",
    },
}


def _get_model(
    experiment_name: str,
    pretrained: bool,
    progress: bool,
    base_network: str = "bcos_final",
    dataset: str = "ImageNet",
    **model_kwargs: Any,
) -> torch.nn.Module:
    """
    Helper that loads the model and attaches its config and
    transform to it as `config` and `transform` respectively.
    """
    # load empty model
    exp = Experiment(dataset, base_network, experiment_name)
    bcos_args = {}
    if "bcos_args" in model_kwargs:
        bcos_args = dict(bcos_args=model_kwargs.pop("bcos_args"))
    model = exp.get_model(args=model_kwargs, **bcos_args)

    # attach stuff
    assert not hasattr(model, "config")
    assert not hasattr(model, "transform")
    model.config = exp.config
    model.transform = model.config["data"]["test_transform"]

    # load weights if needed
    if pretrained:
        url = URLS[base_network][experiment_name]
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            progress=progress,
            check_hash=True,
        )
        model.load_state_dict(state_dict)

    return model


# ------------------------------- [model entrypoints] -------------------------------------
@register
def resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> "torch.nn.Module":
    r"""B-cos ResNet-18.

    B-cos version of a ResNet-18 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  68.736%      |
    +---------+---------------+
    | Acc@5   |  87.430%      |
    +---------+---------------+
    | #Params |  11.69M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnet_18", pretrained, progress, **kwargs)


@register
def resnet34(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-34

    B-cos version of a ResNet-34 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  72.284%      |
    +---------+---------------+
    | Acc@5   |  90.052%      |
    +---------+---------------+
    | #Params |  21.80M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnet_34", pretrained, progress, **kwargs)


@register
def resnet50(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-50

    B-cos version of a ResNet-50 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  75.882%      |
    +---------+---------------+
    | Acc@5   |  92.064%      |
    +---------+---------------+
    | #Params |  25.54M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnet_50", pretrained, progress, **kwargs)


@register
def resnet101(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-101

    B-cos version of a ResNet-101 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  76.532%      |
    +---------+---------------+
    | Acc@5   |  92.538%      |
    +---------+---------------+
    | #Params |  44.50M       |
    +---------+---------------+

    References
    ----------
    .. [1] `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    .. [2] `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnet_101", pretrained, progress, **kwargs)


@register
def resnet152(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-152

    B-cos version of a ResNet-152 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  76.484%      |
    +---------+---------------+
    | Acc@5   |  92.398%      |
    +---------+---------------+
    | #Params |  60.13M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnet_152", pretrained, progress, **kwargs)


@register
def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNeXt-50 32x4d

    B-cos version of a ResNeXt-50 32x4d model.

    +---------+---------------------+
    | Name    | Value               |
    +=========+=====================+
    | Acc@1   |  75.820%            |
    +---------+---------------------+
    | Acc@5   |  91.810%            |
    +---------+---------------------+
    | #Params |  25.00M             |
    +---------+---------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Aggregated Residual Transformations for Deep Neural Networks <https://arxiv.org/pdf/1611.05431.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("resnext_50_32x4d", pretrained, progress, **kwargs)


@register
def densenet121(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-121

    B-cos version of a DenseNet-121 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  73.612%      |
    +---------+---------------+
    | Acc@5   |  91.106%      |
    +---------+---------------+
    | #Params |  7.95M        |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("densenet_121", pretrained, progress, **kwargs)


@register
def densenet161(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-161

    B-cos version of a DenseNet-161 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  76.622%      |
    +---------+---------------+
    | Acc@5   |  92.554%      |
    +---------+---------------+
    | #Params |  28.58M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("densenet_161", pretrained, progress, **kwargs)


@register
def densenet169(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-169

    B-cos version of a DenseNet-169 model.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  75.186%      |
    +---------+---------------+
    | Acc@5   |  91.786%      |
    +---------+---------------+
    | #Params |  14.08M       |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("densenet_169", pretrained, progress, **kwargs)


@register
def densenet201(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-201

    B-cos version of a DenseNet-201 model.

    +---------+-----------------+
    | Name    | Value           |
    +=========+=================+
    | Acc@1   |  75.480%        |
    +---------+-----------------+
    | Acc@5   |  91.992%        |
    +---------+-----------------+
    | #Params |  19.91M         |
    +---------+-----------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("densenet_201", pretrained, progress, **kwargs)


@register
def vgg11_bnu(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos VGG-11 BNU

    B-cos version of a VGG-11 model with Batch Normalization without centering.

    +---------+---------------+
    | Name    | Value         |
    +=========+===============+
    | Acc@1   |  69.310%      |
    +---------+---------------+
    | Acc@5   |  88.388%      |
    +---------+---------------+
    | #Params |  132.86M      |
    +---------+---------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/pdf/1409.1556.pdf>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model("vgg_11_bnu", pretrained, progress, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Models trained much longer (600 epochs) with better accuracies.
# ----------------------------------------------------------------------------------------------------------------------
@register
def convnext_tiny(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ConvNeXt-T

    B-cos version of a ConvNeXt-Tiny model with HW-normalization.

    The weights are from epoch 596's (starting from 0) EMA checkpoint weights.
    The model was trained with AMP.

    +---------+------------------+
    | Name    | Value            |
    +=========+==================+
    | Acc@1   |  77.488%         |
    +---------+------------------+
    | Acc@5   |  93.192%         |
    +---------+------------------+
    | #Params |  28.54M           |
    +---------+------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "convnext_tiny_pn",
        pretrained,
        progress,
        base_network="bcos_final_long",
        **kwargs,
    )


@register
def convnext_base(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ConvNeXt-B

    B-cos version of a ConvNeXt-Base model with HW-normalization.

    The weights are from epoch 581's (starting from 0) non-EMA checkpoint weights.
    The model was trained with AMP.

    +---------+------------------+
    | Name    | Value            |
    +=========+==================+
    | Acc@1   |  79.650%         |
    +---------+------------------+
    | Acc@5   |  94.614%         |
    +---------+------------------+
    | #Params |  88.47M          |
    +---------+------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "convnext_base_pn",
        pretrained,
        progress,
        base_network="bcos_final_long",
        **kwargs,
    )


@register
def convnext_tiny_bnu(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ConvNeXt-T with BNU

    B-cos version of a ConvNeXt-Tiny model with Batch Normalization without centering.

    The weights are from epoch 456's (starting from 0) EMA checkpoint weights.
    The model was trained with AMP.

    +---------+------------------+
    | Name    | Value            |
    +=========+==================+
    | Acc@1   |  76.826%         |
    +---------+------------------+
    | Acc@5   |  93.090%         |
    +---------+------------------+
    | #Params |  28.54M          |
    +---------+------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "convnext_tiny_bnu",
        pretrained,
        progress,
        base_network="bcos_final_long",
        **kwargs,
    )


@register
def convnext_base_bnu(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ConvNeXt-B with BNU

    B-cos version of a ConvNeXt-Base model with Batch Normalization without centering.

    The weights are from epoch 541's (starting from 0) EMA checkpoint weights.
    The model was trained with AMP.

    +---------+------------------+
    | Name    | Value            |
    +=========+==================+
    | Acc@1   |  80.142%         |
    +---------+------------------+
    | Acc@5   |  94.834%         |
    +---------+------------------+
    | #Params |  88.47M          |
    +---------+------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "convnext_base_bnu",
        pretrained,
        progress,
        base_network="bcos_final_long",
        **kwargs,
    )


@register
def densenet121_long(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos DenseNet-121 just trained longer.

    The model architecture is the same as the one from `densenet121`.

    The weights are from epoch 596's (starting from 0) non-EMA checkpoint weights.
    The model was trained with AMP.

    +---------+---------------------+
    | Name    | Value               |
    +=========+=====================+
    | Acc@1   |  77.302%            |
    +---------+---------------------+
    | Acc@5   |  93.234%            |
    +---------+---------------------+
    | #Params |  7.95M              |
    +---------+---------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "densenet_121", pretrained, progress, base_network="bcos_final_long", **kwargs
    )


@register
def resnet50_long(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-50 just trained longer.

    The model architecture is the same as the one from `resnet50`.

    The weights are from epoch 580's (starting from 0) non-EMA checkpoint weights.
    The model was trained with AMP.

    +---------+------------------+
    | Name    | Value            |
    +=========+==================+
    | Acc@1   |  79.468%         |
    +---------+------------------+
    | Acc@5   |  94.452%         |
    +---------+------------------+
    | #Params |  25.54M          |
    +---------+------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "resnet_50", pretrained, progress, base_network="bcos_final_long", **kwargs
    )


@register
def resnet152_long(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos ResNet-152 just trained longer.

    The model architecture is the same as the one from `resnet152`.
    The weights are from epoch 433's (starting from 0) non-EMA checkpoint weights.
    The model was trained with AMP.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  80.144%          |
    +---------+-------------------+
    | Acc@5   |  94.116%          |
    +---------+-------------------+
    | #Params |  60.13M           |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    `TorchVision's new recipe
    <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_model(
        "resnet_152", pretrained, progress, base_network="bcos_final_long", **kwargs
    )


# ----------------------------------------------------------------------------------------------------------------------
# ViT models (both B-cos and non-B-cos)
# non-B-cos i.e. standard models are prefixed with "standard_"
# ----------------------------------------------------------------------------------------------------------------------
def _requires_einops():
    """Checks if einops is installed."""
    try:
        import einops  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "This model requires einops to be installed. "
            "To fix this, run `pip install einops`."
        )


def _get_vit_model(*args, **kwargs):
    """Gets a ViT model, which requires einops. Hence, it checks for it."""
    _requires_einops()
    return _get_model(*args, **kwargs)


@register
def simple_vit_ti_patch16_224(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """B-cos Simple ViT-Ti with 16x16 patch size and 224x224 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  59.960%          |
    +---------+-------------------+
    | Acc@5   |  81.838%          |
    +---------+-------------------+
    | #Params |  5.80M            |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_simple_vit_ti_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def simple_vit_s_patch16_224(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-S with 16x16 patch size and 224x224 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  69.246%          |
    +---------+-------------------+
    | Acc@5   |  88.096%          |
    +---------+-------------------+
    | #Params |  22.28M           |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_simple_vit_s_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def simple_vit_b_patch16_224(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-B with 16x16 patch size and 224x224 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  74.408%          |
    +---------+-------------------+
    | Acc@5   |  91.156%          |
    +---------+-------------------+
    | #Params |  86.90M           |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_simple_vit_b_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def simple_vit_l_patch16_224(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-L with 16x16 patch size and 224x224 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  75.060%          |
    +---------+-------------------+
    | Acc@5   |  91.378%          |
    +---------+-------------------+
    | #Params |  178.79M          |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_simple_vit_l_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def vitc_ti_patch1_14(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-Ti with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  67.260%          |
    +---------+-------------------+
    | Acc@5   |  86.774%          |
    +---------+-------------------+
    | #Params |  5.32M            |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_vitc_ti_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def vitc_s_patch1_14(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-S with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  74.504%          |
    +---------+-------------------+
    | Acc@5   |  91.288%          |
    +---------+-------------------+
    | #Params |  20.88M           |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_vitc_s_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def vitc_b_patch1_14(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-B with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  77.152%          |
    +---------+-------------------+
    | Acc@5   |  92.926%          |
    +---------+-------------------+
    | #Params |  81.37M           |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_vitc_b_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def vitc_l_patch1_14(pretrained: bool = False, progress: bool = True, **kwargs):
    """B-cos Simple ViT-L with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  77.782%          |
    +---------+-------------------+
    | Acc@5   |  92.966%          |
    +---------+-------------------+
    | #Params |  167.44M          |
    +---------+-------------------+

    References
    ----------
    `B-cos Networks: Alignment is All We Need for Interpretability <https://arxiv.org/abs/2205.10268>`_

    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "bcos_vitc_l_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_simple_vit_ti_patch16_224(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-Ti with 16x16 patch size and 224x224 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  70.230%          |
    +---------+-------------------+
    | Acc@5   |  89.380%          |
    +---------+-------------------+
    | #Params |  5.67M            |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "simple_vit_ti_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_simple_vit_s_patch16_224(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-S with 16x16 patch size and 224x224 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  74.470%          |
    +---------+-------------------+
    | Acc@5   |  91.226%          |
    +---------+-------------------+
    | #Params |  21.96M           |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "simple_vit_s_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_simple_vit_b_patch16_224(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-B with 16x16 patch size and 224x224 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  75.300%          |
    +---------+-------------------+
    | Acc@5   |  91.026%          |
    +---------+-------------------+
    | #Params |  86.38M           |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "simple_vit_b_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_simple_vit_l_patch16_224(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-L with 16x16 patch size and 224x224 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  75.710%          |
    +---------+-------------------+
    | Acc@5   |  90.050%          |
    +---------+-------------------+
    | #Params |  178.10M          |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "simple_vit_l_patch16_224",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_vitc_ti_patch1_14(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-Ti with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  72.590%          |
    +---------+-------------------+
    | Acc@5   |  90.788%          |
    +---------+-------------------+
    | #Params |  5.33M            |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "vitc_ti_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_vitc_s_patch1_14(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-S with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  75.756%          |
    +---------+-------------------+
    | Acc@5   |  91.994%          |
    +---------+-------------------+
    | #Params |  20.91M           |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "vitc_s_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_vitc_b_patch1_14(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-B with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  76.790%          |
    +---------+-------------------+
    | Acc@5   |  92.024%          |
    +---------+-------------------+
    | #Params |  81.39M           |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "vitc_b_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )


@register
def standard_vitc_l_patch1_14(
    pretrained: bool = False, progress: bool = True, **kwargs
):
    """Standard Simple ViT-L with a convolutional stem.
    The ViT, after the convolution stem, accepts a 1x1 patch size and 14x14 image size.
    This is NOT a B-cos model.

    +---------+-------------------+
    | Name    | Value             |
    +=========+===================+
    | Acc@1   |  77.866%          |
    +---------+-------------------+
    | Acc@5   |  92.298%          |
    +---------+-------------------+
    | #Params |  167.54M          |
    +---------+-------------------+

    References
    ----------
    `Better plain ViT baselines for ImageNet-1k <https://arxiv.org/abs/2205.01580>`_

    `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
    progress : bool
        If True, displays a progress bar of the download to stderr
    **kwargs : Any, optional
        Additional arguments passed to the model constructor
        Please see source code for details.
    """
    return _get_vit_model(
        "vitc_l_patch1_14",
        pretrained,
        progress,
        base_network="vit_final",
        **kwargs,
    )
