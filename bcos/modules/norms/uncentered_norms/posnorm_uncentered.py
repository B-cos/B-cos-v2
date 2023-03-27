"""
Position Normalization Uncentered (PNU) for 2D inputs

Positional Normalization:
https://github.com/Boyiliee/Positional-Normalization
"""
import torch
from torch import Tensor, nn

from bcos.modules.common import DetachableModule

__all__ = [
    "PositionNormUncentered2d",
]


class PositionNormUncentered2d(nn.LayerNorm, DetachableModule):
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

        # ------------ manual PNU forward pass -------------

        # get stats
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        if self.detach:
            var = var.detach()
        std = (var + self.eps).sqrt()

        x = x / std

        # affine transformation
        if self.weight is not None:
            x = self.weight[None, ..., None, None] * x
        if self.bias is not None:
            x = x + self.bias[None, ..., None, None]

        return x
