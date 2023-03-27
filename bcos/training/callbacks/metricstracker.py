from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torchmetrics
from pytorch_lightning.utilities import rank_zero_only


class MetricsTracker(pl_callbacks.Callback):
    def __init__(self):
        self.metrics = None
        self.data = None

    @rank_zero_only
    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self.metrics = {
            name: value
            for name, value in pl_module.named_children()
            if isinstance(value, torchmetrics.Metric)
        }
        self.data = {name: [] for name in self.metrics.keys()}

    @rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.sanity_checking:
            return
        for name in self.metrics.keys():
            value = float(self.metrics[name].compute())
            self.data[name].append((trainer.current_epoch, value))

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        save_dir = Path(trainer.default_root_dir) / "metrics"
        save_dir.mkdir(exist_ok=True)

        # easier to load this way quickly for plotting etc.
        for name, values in self.data.items():
            np.savetxt(save_dir / f"{name}.gz", values)

    # easier to save this way to resume training etc.
    def state_dict(self):
        if self.data is None:
            return {}
        return self.data.copy()

    def load_state_dict(self, state_dict):
        if self.data is None:
            self.data = state_dict.copy()
        else:
            self.data.update(state_dict)
