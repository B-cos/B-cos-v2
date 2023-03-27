"""
B-cos VGG models

Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb8a2/torchvision/models/vgg.py
"""
import math
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from bcos.common import BcosUtilMixin
from bcos.modules import BcosConv2d, LogitLayer, norms

__all__ = [
    "BcosVGG",
    "vgg11",
    "vgg11_bnu",
    "vgg13",
    "vgg13_bnu",
    "vgg16",
    "vgg16_bnu",
    "vgg19_bnu",
    "vgg19",
]


DEFAULT_CONV_LAYER = BcosConv2d
DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)


class BcosVGG(BcosUtilMixin, nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        conv_layer: Callable[..., nn.Module] = None,
        logit_bias: Optional[float] = None,
        logit_temperature: Optional[float] = None,
    ) -> None:
        super(BcosVGG, self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            conv_layer(512, 4096, kernel_size=7, padding=3, scale_fact=1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            conv_layer(4096, 4096, scale_fact=1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            conv_layer(4096, num_classes, scale_fact=1000),
        )
        self.num_classes = num_classes
        self.logit_layer = LogitLayer(
            logit_temperature=logit_temperature,
            logit_bias=logit_bias or -math.log(num_classes - 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)[..., None, None]
        x = self.classifier(x)
        # because it's supposed to come after
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.logit_layer(x)
        return x

    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model. Note this comes before global pooling."""
        return self.classifier

    def get_feature_extractor(self) -> nn.Module:
        """Returns the feature extractor part of the model. Without global pooling."""
        return self.features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(
    cfg: List[Union[str, int]],
    norm_layer: Callable[..., nn.Module] = DEFAULT_NORM_LAYER,
    conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
    in_channels: int = 6,
    no_pool=False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    new_config = []
    for idx, entry in enumerate(cfg):
        new_config.append([entry, 1])
        if entry == "M" and no_pool:
            new_config[idx - 1][1] = 2

    for v, stride in new_config:
        if v == "M":
            if no_pool:
                continue
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = conv_layer(
                in_channels, v, kernel_size=3, padding=1, stride=stride, scale_fact=1000
            )
            if not isinstance(norm_layer, nn.Identity):
                layers += [conv2d, norm_layer(v)]  # , nn.ReLU(inplace=True)]
            else:
                layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(
    arch: str,
    cfg: str,
    pretrained: bool,
    progress: bool,
    norm_layer: Callable[..., nn.Module] = None,
    conv_layer: Callable[..., nn.Module] = None,
    in_chans: int = 6,
    no_pool=False,
    **kwargs: Any
) -> BcosVGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = BcosVGG(
        make_layers(
            cfgs[cfg],
            norm_layer=norm_layer,
            conv_layer=conv_layer,
            in_channels=in_chans,
            no_pool=no_pool,
        ),
        conv_layer=conv_layer,
        **kwargs,
    )
    if pretrained:
        raise ValueError(
            "If you want to load pretrained weights, then please use the entrypoints in "
            "bcos.pretrained or bcos.model.pretrained instead."
        )
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    return _vgg("vgg11", "A", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg11_bnu(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosVGG:
    return _vgg("vgg11_bnu", "A", pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    return _vgg("vgg13", "B", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg13_bnu(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosVGG:
    return _vgg("vgg13_bnu", "B", pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    return _vgg("vgg16", "D", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg16_bnu(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosVGG:
    return _vgg("vgg16_bnu", "D", pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BcosVGG:
    return _vgg("vgg19", "E", pretrained, progress, norm_layer=nn.Identity, **kwargs)


def vgg19_bnu(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosVGG:
    return _vgg("vgg19_bnu", "E", pretrained, progress, **kwargs)
