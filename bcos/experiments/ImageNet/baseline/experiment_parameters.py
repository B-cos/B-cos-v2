"""
For reproducing numbers.
"""
import math  # noqa

import torch.nn as nn

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000

# The recipes have been taken from:
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 64  # per GPU! * 4 = 256 effective
DEFAULT_NUM_EPOCHS = 90
DEFAULT_LR = 0.1
DEFAULT_CROP_SIZE = 224

# no norm layer specification because the models set BN as default anyways

DEFAULT_OPTIMIZER = OptimizerFactory(
    name="SGD",
    lr=DEFAULT_LR,
    momentum=0.9,
    weight_decay=1e-4,
)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="steplr",
    step_size=30,
    gamma=0.1,
)

DEFAULTS = dict(
    data=dict(
        train_transform=ImageNetClassificationPresetTrain(
            crop_size=DEFAULT_CROP_SIZE,
        ),
        test_transform=ImageNetClassificationPresetEval(
            crop_size=DEFAULT_CROP_SIZE,
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=8,
        num_classes=NUM_CLASSES,
    ),
    model=dict(
        is_bcos=False,
        # "name": None,
        args=dict(
            num_classes=NUM_CLASSES,
        ),
    ),
    criterion=nn.CrossEntropyLoss(),
    test_criterion=nn.CrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
)


# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


RESNET_DEPTHS = [18, 34, 50, 101, 152]
resnets = {
    f"resnet_{depth}": update_default(
        dict(
            model=dict(
                name=f"resnet{depth}",
            ),
        )
    )
    for depth in RESNET_DEPTHS
}

DENSENET_DEPTHS = [121, 161, 201, 169]
densenets = {
    f"densenet_{depth}": update_default(
        dict(
            model=dict(
                name=f"densenet{depth}",
            ),
        )
    )
    for depth in DENSENET_DEPTHS
}

vggs = {
    "vgg_11_bn": update_default(
        dict(
            model=dict(
                name="vgg11_bn",
            ),
        )
    ),
}

# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(resnets)
CONFIGS.update(densenets)
CONFIGS.update(vggs)
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[420, 1337]))


if __name__ == "__main__":
    configs_cli(CONFIGS)
