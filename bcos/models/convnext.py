"""
B-cos ConvNeXt models

Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb/torchvision/models/convnext.py
"""
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torchvision.ops import StochasticDepth

from bcos.common import BcosUtilMixin
from bcos.modules import BcosConv2d, LogitLayer, norms

__all__ = [
    "BcosConvNeXt",
    "convnext_atto",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "CNBlock",
    "CNBlockConfig",
]

DEFAULT_NORM_LAYER = norms.NoBias(norms.DetachablePositionNorm2d)
DEFAULT_CONV_LAYER = BcosConv2d


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(DEFAULT_NORM_LAYER, eps=1e-6)

        self.block = nn.Sequential(
            conv_layer(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False),
            # Permute([0, 2, 3, 1]),
            norm_layer(dim),
            conv_layer(
                in_channels=dim, out_channels=4 * dim, kernel_size=1, bias=False
            ),
            # nn.GELU(),  # The B-cos transform itself is non-linear
            conv_layer(
                in_channels=4 * dim, out_channels=dim, kernel_size=1, bias=False
            ),
            # Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class BcosConvNeXt(BcosUtilMixin, nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        in_chans: int = 6,
        block: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        logit_bias: Optional[float] = None,
        logit_temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(DEFAULT_NORM_LAYER, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.extend(
            [
                conv_layer(
                    in_chans,
                    firstconv_output_channels,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=False,
                ),
                norm_layer(firstconv_output_channels),
            ]
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(
                    block(
                        cnf.input_channels,
                        layer_scale,
                        sd_prob,
                        conv_layer=conv_layer,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        conv_layer(
                            cnf.input_channels,
                            cnf.out_channels,
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            conv_layer(
                lastconv_output_channels, num_classes, kernel_size=1, bias=False
            ),
        )
        self.num_classes = num_classes
        self.logit_layer = LogitLayer(
            logit_temperature=logit_temperature,
            logit_bias=logit_bias or -math.log(num_classes - 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.logit_layer(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model. Note this comes before global pooling."""
        return self.classifier

    def get_feature_extractor(self) -> nn.Module:
        """Returns the feature extractor part of the model. Without global pooling."""
        return self.features


def _convnext(
    arch: str,
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> BcosConvNeXt:
    model = BcosConvNeXt(
        block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs
    )

    if pretrained:
        raise ValueError(
            "If you want to load pretrained weights, then please use the entrypoints in "
            "bcos.pretrained or bcos.model.pretrained instead."
        )

    return model


def convnext_atto(
    *, pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosConvNeXt:
    block_setting = [
        CNBlockConfig(40, 80, 2),
        CNBlockConfig(80, 160, 2),
        CNBlockConfig(160, 320, 6),
        CNBlockConfig(320, None, 2),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(
        "convnext_atto",
        block_setting,
        stochastic_depth_prob,
        pretrained,
        progress,
        **kwargs,
    )


def convnext_tiny(
    *, pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(
        "convnext_tiny",
        block_setting,
        stochastic_depth_prob,
        pretrained,
        progress,
        **kwargs,
    )


def convnext_small(
    *, pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(
        "convnext_small",
        block_setting,
        stochastic_depth_prob,
        pretrained,
        progress,
        **kwargs,
    )


def convnext_base(
    *, pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosConvNeXt:
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(
        "convnext_base",
        block_setting,
        stochastic_depth_prob,
        pretrained,
        progress,
        **kwargs,
    )


def convnext_large(
    *, pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosConvNeXt:
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(
        "convnext_large",
        block_setting,
        stochastic_depth_prob,
        pretrained,
        progress,
        **kwargs,
    )
