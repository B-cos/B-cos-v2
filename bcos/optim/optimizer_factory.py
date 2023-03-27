"""
Nothing fancy, just for creating optimizers.

Modified from: https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf054/references/classification/train.py
"""
import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch

__all__ = ["OptimizerFactory"]


class OptimizerFactory:
    def __init__(
        self,
        name: str,
        lr: Union[int, float],
        **kwargs,
    ):
        self.name = name
        self.args = dict(
            lr=lr,
            **kwargs,
        )

        assert name.lower() in ["adamw", "adam", "sgd", "rmsprop"]

    def create(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Creates an optimizer with the preset configuration.
        Parameters
        ----------
        model: torch.nn.Module
            The model to optimize.

        Returns
        -------

        """
        name = self.name.lower()
        args = self.args.copy()

        if "weight_decay" in self.args or name == "adamw":
            warnings.warn(
                "Just FYI: Setting a different WD for "
                "**uncentered** norms is not configured! "
                "You can safely ignore this for centered ones though."
            )
            parameters = self.set_weight_decay(model, self.args["weight_decay"])
        else:
            parameters = model.parameters()

        if name == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                **args,
            )
        elif name == "rmsprop":
            args["eps"] = args.get("eps", 0.0316)
            args["alpha"] = args.get("alpha", 0.9)
            optimizer = torch.optim.RMSprop(
                parameters,
                **args,
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(parameters, **args)
        elif name == "adam":
            if args.get("weight_decay", 0) != 0:
                warnings.warn(
                    "You probably should be using AdamW instead of Adam for WD!"
                )
            optimizer = torch.optim.Adam(parameters, **args)
        else:
            raise ValueError(
                f"Invalid optimizer '{name}'. Only SGD, RMSprop, Adam and AdamW are supported."
            )

        return optimizer

    # with methods
    def with_name(self, new_name):
        """
        Creates a new optimizer factory with same parameters
        except with the new name.
        """
        args = copy.deepcopy(self.args)
        lr = args.pop("lr")
        new_optimizer_factory = type(self)(
            name=new_name,
            lr=lr,
            **args,
        )

        return new_optimizer_factory

    def with_lr(self, lr):
        """
        Creates a new optimizer factory with same parameters
        except with the new lr.
        """
        args = copy.deepcopy(self.args)
        del args["lr"]
        new_optimizer_factory = type(self)(
            name=self.name,
            lr=lr,
            **args,
        )

        return new_optimizer_factory

    def with_args(self, **kwargs):
        assert "lr" not in kwargs, "lr should not be in given args, use with_lr instead"
        assert (
            "name" not in kwargs
        ), "name should not be in given args, use with_name instead"

        new_optimizer_factory = type(self)(
            name=self.name,
            lr=self.args["lr"],
            **kwargs,
        )

        return new_optimizer_factory

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(optimizer='{self.name}'"
        if self.args:
            for k, v in self.args.items():
                s += f", {k}={v}"
        s += ")"

        return s

    def __to_config__(self):
        """See bcos.experiments.utils.sanitize_config for details."""
        return dict(
            name=self.name.lower(),
            **self.args,
        )

    @staticmethod
    def set_weight_decay(
        model: torch.nn.Module,
        weight_decay: float,
        norm_weight_decay: Optional[float] = None,
        norm_classes: Optional[List[type]] = None,
        custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
    ):
        """
        From torchvision reference.
        """
        if not norm_classes:
            norm_classes = [
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.LayerNorm,
                torch.nn.GroupNorm,
                torch.nn.modules.instancenorm._InstanceNorm,
                torch.nn.LocalResponseNorm,
            ]
        norm_classes = tuple(norm_classes)

        params = {
            "other": [],
            "norm": [],
        }
        params_weight_decay = {
            "other": weight_decay,
            "norm": norm_weight_decay,
        }
        custom_keys = []
        if custom_keys_weight_decay is not None:
            for key, weight_decay in custom_keys_weight_decay:
                params[key] = []
                params_weight_decay[key] = weight_decay
                custom_keys.append(key)

        def _add_params(module, prefix=""):
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                is_custom_key = False
                for key in custom_keys:
                    target_name = (
                        f"{prefix}.{name}" if prefix != "" and "." in key else name
                    )
                    if key == target_name:
                        params[key].append(p)
                        is_custom_key = True
                        break
                if not is_custom_key:
                    if norm_weight_decay is not None and isinstance(
                        module, norm_classes
                    ):
                        params["norm"].append(p)
                    else:
                        params["other"].append(p)

            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
                _add_params(child_module, prefix=child_prefix)

        _add_params(model)

        param_groups = []
        for key in params:
            if len(params[key]) > 0:
                param_groups.append(
                    {"params": params[key], "weight_decay": params_weight_decay[key]}
                )
        return param_groups
