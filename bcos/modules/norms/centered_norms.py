"""
Centered Norms.

Code partially taken from
https://github.com/pytorch/pytorch/blob/9e81c0c3f46a36333e82b799b4afa79b44b6bb59/torch/nn/modules/batchnorm.py

Position Norm implementation:
https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py

Positional Normalization:
https://github.com/Boyiliee/Positional-Normalization

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bcos.modules.common import DetachableModule

__all__ = [
    "AllNorm2d",
    "BatchNorm2d",
    "DetachableGroupNorm2d",
    "DetachableGNInstanceNorm2d",
    "DetachableGNLayerNorm2d",
    "DetachableLayerNorm",
    "DetachablePositionNorm2d",
]


# The easiest way is to use BN3D
class AllNorm2d(nn.BatchNorm3d):
    """
    The AllNorm.
    """

    def __init__(
        self,
        num_features: int,
        *args,
        **kwargs,
    ) -> None:
        # since we do it over the whole thing we have to set
        # this to one
        super().__init__(
            1,
            *args,
            **kwargs,
        )

    def forward(self, input: "Tensor") -> "Tensor":
        original_shape = input.shape
        # (B,C,H,W) -> (B,1,C,H,W)
        input = input.unsqueeze(1)

        # (B,1,C,H,W) normed
        output = super().forward(input)

        # (B,C,H,W) normed
        return output.reshape(original_shape)

    def set_explanation_mode(self, activate: bool = True):
        if activate:
            assert (
                not self.training
            ), "Centered AllNorm only supports explanation mode during .eval()!"


# just for a warnable version
class BatchNorm2d(nn.BatchNorm2d):
    def set_explanation_mode(self, activate: bool = True):
        if activate:
            assert (
                not self.training
            ), "Centered BN only supports explanation mode during .eval()!"


class DetachableGroupNorm2d(nn.GroupNorm, DetachableModule):
    def __init__(self, *args, **kwargs):
        DetachableModule.__init__(self)
        super().__init__(*args, **kwargs)

    def forward(self, input: "Tensor") -> "Tensor":
        # input validation
        assert input.dim() == 4, f"Expected 4D input got {input.dim()}D instead!"
        assert input.shape[1] % self.num_groups == 0, (
            "Number of channels in input should be divisible by num_groups, "
            f"but got input of shape {input.shape} and num_groups={self.num_groups}"
        )

        # use faster version if possible
        if not self.detach:
            return F.group_norm(
                input, self.num_groups, self.weight, self.bias, self.eps
            )

        # ------------ manual GN forward pass -------------
        # separate the groups
        # (N, C, *) -> (N, G, C // G, *)
        N, C = input.shape[:2]
        x = input.reshape(N, self.num_groups, C // self.num_groups, *input.shape[2:])

        # calc stats
        var, mean = torch.var_mean(
            x, dim=tuple(range(2, x.dim())), unbiased=False, keepdim=True
        )
        var = var.detach()
        std = (var + self.eps).sqrt()

        # normalize
        x = (x - mean) / std

        # reshape back
        x = x.reshape(input.shape)

        # affine transformation
        if self.weight is not None:
            x = self.weight[None, ..., None, None] * x

        if self.bias is not None:
            x = x + self.bias[None, ..., None, None]

        return x


class DetachableGNInstanceNorm2d(DetachableGroupNorm2d):
    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=num_channels,
            num_channels=num_channels,
            *args,
            **kwargs,
        )


class DetachableGNLayerNorm2d(DetachableGroupNorm2d):
    """
    A CNN detachable layer norm.
    """

    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(
            num_groups=1,
            num_channels=num_channels,
            *args,
            **kwargs,
        )


class DetachableLayerNorm(nn.LayerNorm, DetachableModule):
    """
    A non-CNN detachable Layer Norm.
    This is used for the transformers!
    """

    def __init__(self, *args, **kwargs):
        DetachableModule.__init__(self)
        super().__init__(*args, **kwargs)

    def forward(self, input: "Tensor") -> "Tensor":
        # if not detaching -> just use normal pytorch forward pass
        if not self.detach:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # ------------ manual LN detached forward pass -------------
        d_num = len(self.normalized_shape)

        # calc stats
        var, mean = torch.var_mean(
            input, dim=tuple(range(-d_num, 0)), unbiased=False, keepdim=True
        )
        var = var.detach()
        std = (var + self.eps).sqrt_()

        # normalize
        x = (input - mean) / std

        # affine transformation
        if self.weight is not None:
            x = self.weight * x

        if self.bias is not None:
            x = x + self.bias

        return x


class DetachablePositionNorm2d(nn.LayerNorm, DetachableModule):
    def __init__(
        self,
        features: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        assert isinstance(
            features, int
        ), f"Provide #features as an int not {type(features)=}"
        DetachableModule.__init__(self)
        super().__init__(
            normalized_shape=features,
            eps=eps,
            elementwise_affine=affine,
            device=device,
            dtype=dtype,
        )
        self.features = features

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 4, f"input should be 4D not {x.dim()}D"

        # use faster version if possible
        if not self.detach:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

        # ------------ manual PN detached forward pass -------------
        # get stats
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        var = var.detach()
        std = (var + self.eps).sqrt()

        # normalize
        x = (x - mean) / std

        # affine transformation
        if self.weight is not None:
            x = self.weight[None, ..., None, None] * x
        if self.bias is not None:
            x = x + self.bias[None, ..., None, None]

        return x
