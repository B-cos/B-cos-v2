from typing import Optional

import torch.nn as nn
from torch import Tensor

__all__ = [
    "LogitLayer",
]


class LogitLayer(nn.Module):
    def __init__(
        self,
        logit_temperature: Optional[float] = None,
        logit_bias: Optional[float] = None,
    ):
        # note: T=None => T=1 and b=None => b=0
        super().__init__()
        self.logit_bias = logit_bias
        self.logit_temperature = logit_temperature

    def forward(self, in_tensor: Tensor) -> Tensor:
        if self.logit_temperature is not None:
            in_tensor = in_tensor / self.logit_temperature
        if self.logit_bias is not None:
            in_tensor = in_tensor + self.logit_bias
        return in_tensor

    def extra_repr(self) -> str:
        ret = ""
        if self.logit_temperature is not None:
            ret += f"logit_temperature={self.logit_temperature}, "
        if self.logit_bias is not None:
            ret += f"logit_bias={self.logit_bias}, "
        ret = ret[:-2]
        return ret
