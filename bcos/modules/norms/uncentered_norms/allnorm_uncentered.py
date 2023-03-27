"""
All norm without centering.

Code partially taken from
https://github.com/pytorch/pytorch/blob/9e81c0c3f46a36333e82b799b4afa79b44b6bb59/torch/nn/modules/batchnorm.py
"""
from typing import Optional

import torch.nn as nn
from torch import Tensor

from bcos.modules.common import DetachableModule

__all__ = [
    "AllNormUncentered2d",
]


def all_norm_uncentered_2d(
    input: Tensor,
    running_var: Optional[Tensor],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    detach: bool = False,
):
    """
    Uncentered AllNorm. Accepts only batched color image tensors.
    """
    assert input.dim() == 4, "input should be a 4d tensor!"

    if training:
        # first calc stats
        x = input.detach() if detach else input
        var = x.var(unbiased=False)

        # update running stats if given
        if running_var is not None:
            running_var.copy_((1 - momentum) * running_var + momentum * var.detach())

    else:  # evaluation mode
        assert running_var is not None, "running_var must be defined in eval mode"
        var = running_var

    std = (var + eps).sqrt()[None, ..., None, None]

    result = input / std

    if weight is not None:
        result = weight[None, ..., None, None] * result
    if bias is not None:
        result = result + bias[None, ..., None, None]

    return result


class AllNormUncentered2d(nn.BatchNorm2d, DetachableModule):
    def __init__(
        self,
        num_features: int,
        *args,
        **kwargs,
    ) -> None:
        DetachableModule.__init__(self)

        # since we do it over the whole thing we have to set
        # this to one
        super().__init__(
            1,
            *args,
            **kwargs,
        )

    def forward(self, input):
        self._check_input_dim(input)  # require 4

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return all_norm_uncentered_2d(
            input=input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_var=self.running_var
            if not self.training or self.track_running_stats
            else None,
            weight=self.weight,
            bias=self.bias,
            training=bn_training,
            momentum=exponential_average_factor,
            eps=self.eps,
            detach=self.detach,
        )
