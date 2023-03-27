"""
Adaptive Gradient Clipping (https://arxiv.org/abs/2102.06171)

Code references:
official: https://github.com/deepmind/deepmind-research/tree/master/nfnets
timm: https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6/timm/utils/agc.py
lucidrains: https://gist.github.com/lucidrains/0d6560077edac419ab5d3aa29e674d5c
"""
import torch


def unitwise_norm(x: torch.Tensor, norm_type: float = 2.0) -> torch.Tensor:
    if x.squeeze().ndim <= 1:
        dim = None
        keepdim = False
    elif x.ndim in (2, 3):
        dim = 1
        keepdim = True
    elif x.ndim == 4:  # OIHW
        dim = (1, 2, 3)
        keepdim = True
    else:
        raise ValueError(f"Expected 1 <= x.ndim <= 4. Got {x.ndim=}")

    return x.norm(norm_type, dim=dim, keepdim=keepdim)


def adaptive_clip_grad_(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = (
            unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        )
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)
