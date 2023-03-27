from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
import torch

from ..exceptions import EMANotFound, MetricsNotFoundError

__all__ = [
    "Metrics",
]

PathLike = Union[str, Path]


class Metrics(dict):
    """
    A dictionary for storing metrics with some additional helper methods.
    """

    VALIDATION_KEY = "eval_acc1"
    """The key for the validation accuracy."""
    EMA_VALIDATION_KEY = "eval_acc1_ema"
    """The key for the EMA validation accuracy."""

    def __init__(self, other: dict):
        metrics = {name: torch.tensor(value) for name, value in other.items()}
        super().__init__(metrics)

    @classmethod
    def from_metrics_dir(cls, metrics_dir: PathLike) -> "Metrics":
        """
        Loads metrics from a directory.

        Parameters
        ----------
        metrics_dir : PathLike
            The metrics directory to load metrics from.

        Returns
        -------
        Metrics
            The loaded metrics.
        """
        metrics_dir = Path(metrics_dir)
        if not metrics_dir.exists():
            raise MetricsNotFoundError(
                f"Metrics directory '{metrics_dir}' does not exist!"
            )
        metrics = {}
        for metric_file in metrics_dir.glob("*.gz"):
            metric_name = metric_file.stem
            metric_values = np.loadtxt(metric_file)
            metrics[metric_name] = metric_values
        return cls(metrics)

    @classmethod
    def from_experiment_dir(cls, exp_dir: PathLike) -> "Metrics":
        """
        Loads metrics from an experiment directory.

        Parameters
        ----------
        exp_dir : PathLike
            The experiment directory to load metrics from.

        Returns
        -------
        Metrics
            The loaded metrics.
        """
        exp_dir = Path(exp_dir)
        if not exp_dir.exists():
            raise MetricsNotFoundError(
                f"Experiment directory '{exp_dir}' does not exist!"
            )

        metrics_dir = exp_dir / "metrics"
        metrics = cls.from_metrics_dir(metrics_dir)
        return metrics

    def get_best_epoch_and_accuracy(self) -> Tuple[int, float]:
        """
        Gets the best epoch and accuracy w.r.t. the top-1 validation accuracy.

        Returns
        -------
        Tuple[int, float]
            The best epoch and accuracy, respectively.
        """
        return self.find_best_epoch_and_metric_value_for(self.VALIDATION_KEY)

    def get_best_epoch_and_accuracy_ema(self) -> Tuple[int, float]:
        """
        Gets the best EMA epoch and accuracy w.r.t. the top-1 validation accuracy.

        Returns
        -------
        Tuple[int, float]
            The best EMA epoch and accuracy, respectively.
        """
        if self.EMA_VALIDATION_KEY not in self:
            raise EMANotFound("EMA metrics not found!")

        return self.find_best_epoch_and_metric_value_for(self.EMA_VALIDATION_KEY)

    def find_best_epoch_and_metric_value_for(
        self,
        metric_key: str,
        mode: Literal["min", "max"] = "max",
    ) -> Tuple[int, float]:
        """
        Finds the best epoch and metric value in metric collection
        for given metric key/name.

        Parameters
        ----------
        metric_key : str
            The metric key/name to search in.
        mode : Literal["min", "max"]
            Whether to min. or max. to find best. Default: max.

        Returns
        -------
        Tuple[int, float]
            The epoch and best metric value, respectively.
        """

        try:
            metric_values = self[metric_key][:, 1]
        except KeyError:
            if metric_key == self.EMA_VALIDATION_KEY:
                raise EMANotFound()
            else:
                raise
        if mode == "max":
            best_epoch_idx = metric_values.argmax()
        elif mode == "min":
            best_epoch_idx = metric_values.argmin()
        else:
            raise ValueError(f"Unknown {mode=}")

        best_entry = self[metric_key][best_epoch_idx]
        best_epoch = int(best_entry[0])
        best_metric_value = float(best_entry[1])
        return best_epoch, best_metric_value
