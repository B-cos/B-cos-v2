"""
Nothing fancy, just for creating lr schedulers.

Modified from: https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf054/references/classification/train.py
"""
from typing import Literal, Optional

import torch

__all__ = ["LRSchedulerFactory"]


class LRSchedulerFactory:
    def __init__(
        self,
        name: str,
        step_size: int = 30,  # interpreted as epochs, converted to steps if needed
        gamma: float = 0.1,
        epochs: Optional[int] = None,
        lr_min: float = 0.0,
        interval: Literal["epoch", "step"] = "epoch",
        # warmup
        warmup_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_method: str = "constant",
        warmup_decay: float = 0.01,
    ):
        self.name = name
        self.step_size = step_size
        self.gamma = gamma
        self.epochs = epochs
        self.lr_min = lr_min

        self.interval = interval

        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        self.warmup_decay = warmup_decay

        assert name.lower() in ["steplr", "cosineannealinglr", "exponentiallr"]
        assert warmup_method.lower() in ["linear", "constant"]
        assert interval in ["epoch", "step"]
        assert name.lower() != "cosineannealinglr" or (
            self.epochs is not None and self.epochs > 0
        ), "If name='cosineannealinglr' then (positive) epochs need to be provided!"
        assert interval == "epoch" or (
            self.epochs is not None and self.epochs > 0
        ), "If interval='step' then (positive) epochs need to be provided!"
        assert (
            warmup_steps is None or warmup_epochs is None
        ), "Do not provide both warmup_steps and warmup_epochs at the same time!"
        assert (
            interval == "step" or self.warmup_steps is None
        ), "For warmup_steps, interval must be 'step'"
        assert (
            warmup_epochs is None or warmup_epochs > 0
        ), "Provided positive warmup_epochs, if then"
        assert (
            warmup_steps is None or warmup_steps > 0
        ), "Provided positive warmup_steps, if then"

    def create(
        self, optimizer: torch.optim.Optimizer, total_steps: Optional[int] = None
    ):
        assert (
            total_steps is not None or self.interval == "epoch"
        ), "interval='step' requires total_steps!"

        name = self.name.lower()
        if name == "steplr":
            step_size = self.step_size
            if self.interval == "step":
                num_steps_per_epoch = total_steps // self.epochs
                step_size *= num_steps_per_epoch
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=self.gamma
            )
        elif name == "cosineannealinglr":
            T_max = self.epochs
            if self.warmup_epochs is not None:
                T_max -= self.warmup_epochs
            if self.interval == "step":
                num_steps_per_epoch = total_steps // self.epochs
                T_max *= num_steps_per_epoch
                if self.warmup_steps is not None:
                    assert self.warmup_epochs is None
                    T_max -= self.warmup_steps

            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.lr_min
            )
        elif name == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.gamma
            )
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{name}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if self.warmup_epochs or self.warmup_steps:
            warmup_method = self.warmup_method.lower()
            if self.interval == "epoch":
                total_iters = self.warmup_epochs
            else:
                num_steps_per_epoch = total_steps // self.epochs
                if self.warmup_steps is not None:
                    total_iters = self.warmup_steps
                else:
                    total_iters = self.warmup_epochs * num_steps_per_epoch

            if warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.warmup_decay,
                    total_iters=total_iters,
                )
            elif warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=self.warmup_decay,
                    total_iters=total_iters,
                )
            else:
                raise ValueError(
                    f"Invalid warmup lr method '{self.warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[total_iters],
            )
        else:
            lr_scheduler = main_lr_scheduler

        if self.interval == "epoch":
            return lr_scheduler

        return dict(
            scheduler=lr_scheduler,
            interval="step",
        )

    def with_epochs(self, epochs: Optional[int]):
        return type(self)(
            name=self.name,
            step_size=self.step_size,
            gamma=self.step_size,
            epochs=epochs,
            lr_min=self.lr_min,
            interval=self.interval,
            warmup_epochs=self.warmup_epochs,
            warmup_decay=self.warmup_decay,
            warmup_method=self.warmup_method,
        )

    def __repr__(self):
        s = self.__class__.__name__
        name = self.name.lower()
        s += f"(scheduler='{name}'"

        args = dict()
        if name == "steplr":
            args = dict(step_size=self.step_size, gamma=self.gamma)
        elif name == "cosineannealinglr":
            args = dict(epochs=self.epochs, lr_min=self.lr_min)
        elif name == "exponentiallr":
            args = dict(gamma=self.gamma)

        if self.warmup_epochs or self.warmup_steps:
            args.update(
                dict(
                    warmup_method=self.warmup_method,
                    warmup_epochs=self.warmup_epochs,
                    warmup_steps=self.warmup_steps,
                    warmup_decay=self.warmup_decay,
                )
            )

        args.update(dict(interval=self.interval))

        if args:
            for k, v in args.items():
                s += f", {k}={v}"
        s += ")"

        return s

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        name = self.name.lower()
        result = dict(name=name)

        if name == "steplr":
            result.update(dict(step_size=self.step_size, gamma=self.gamma))
        elif name == "cosineannealinglr":
            result.update(dict(epochs=self.epochs, lr_min=self.lr_min))
        elif name == "exponentiallr":
            result.update(dict(gamma=self.gamma))

        if self.warmup_epochs or self.warmup_steps:
            result.update(
                dict(
                    warmup_method=self.warmup_method,
                    warmup_epochs=self.warmup_epochs,
                    warmup_steps=self.warmup_steps,
                    warmup_decay=self.warmup_decay,
                )
            )

        result.update(dict(interval=self.interval))

        return result
