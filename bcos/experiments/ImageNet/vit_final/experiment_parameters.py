"""
Configs for ViTs, both B-cos and non-B-cos (standard).

Paper: https://arxiv.org/abs/2205.01580
"""
import math  # noqa

from torch import nn

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import configs_cli, update_config
from bcos.modules import DetachableGNLayerNorm2d, norms
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000

DEFAULT_BATCH_SIZE = 128  # per GPU! * 8 = 1024 effective
DEFAULT_NUM_EPOCHS = 90
DEFAULT_LR = 1e-3
DEFAULT_CROP_SIZE = 224


DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=10_000,
    interval="step",
    warmup_decay=0.01,
)

LONG_WARM_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_steps=50_000,
    interval="step",
    warmup_decay=0.01,
)

DEFAULTS = dict(
    data=dict(
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,
        mixup_alpha=0.2,
    ),
    model=dict(
        args=dict(
            num_classes=NUM_CLASSES,
        ),
    ),
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    use_agc=True,
)


# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


def is_big_model(model_name: str) -> bool:
    return "_l_" in model_name or "simple_vit_b" in model_name


SIMPLE_VIT_ARCHS = [
    "simple_vit_ti_patch16_224",
    "simple_vit_s_patch16_224",
    "simple_vit_b_patch16_224",
    "simple_vit_l_patch16_224",
    "vitc_s_patch1_14",
    "vitc_ti_patch1_14",
    "vitc_b_patch1_14",
    "vitc_l_patch1_14",
]

baseline = {
    f"{name}": update_default(
        dict(
            data=dict(
                batch_size=DEFAULT_BATCH_SIZE
                if not is_big_model(name)
                else DEFAULT_BATCH_SIZE // 2,
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    auto_augment_policy="ra",
                    ra_magnitude=10,
                    is_bcos=False,
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=False,
                ),
            ),
            model=dict(
                is_bcos=False,
                name=name,
                args=dict(
                    # linear_layer and conv2d_layer set by model.py
                    norm_layer=nn.LayerNorm,
                    norm2d_layer=DetachableGNLayerNorm2d,
                    act_layer=nn.GELU,
                    channels=3,
                ),
            ),
            criterion=nn.CrossEntropyLoss(),
            test_criterion=nn.CrossEntropyLoss(),
            optimizer=OptimizerFactory(
                "AdamW",
                lr=DEFAULT_LR,
                weight_decay=0.0001,
            ),
            use_agc=False,
            lr_scheduler=DEFAULT_LR_SCHEDULE
            if not is_big_model(name)
            else LONG_WARM_SCHEDULE,
            trainer=dict(
                gradient_clip_val=1.0,
            ),
        )
    )
    for name in SIMPLE_VIT_ARCHS
}


bcos = {
    f"bcos_{name}": update_default(
        dict(
            data=dict(
                batch_size=DEFAULT_BATCH_SIZE
                if not is_big_model(name)
                else DEFAULT_BATCH_SIZE // 2,
                train_transform=ImageNetClassificationPresetTrain(
                    crop_size=DEFAULT_CROP_SIZE,
                    auto_augment_policy="ra",
                    ra_magnitude=10,
                    is_bcos=True,
                ),
                test_transform=ImageNetClassificationPresetEval(
                    crop_size=DEFAULT_CROP_SIZE,
                    is_bcos=True,
                ),
                num_workers=10,
            ),
            model=dict(
                is_bcos=True,
                name=name,
                args=dict(
                    # linear_layer and conv2d_layer set by model.py
                    norm_layer=norms.NoBias(norms.DetachableLayerNorm),
                    act_layer=nn.Identity,
                    channels=6,
                    norm2d_layer=norms.NoBias(DetachableGNLayerNorm2d),
                ),
                bcos_args=dict(
                    b=2,
                    max_out=1,
                ),
                logit_bias=math.log(1 / (NUM_CLASSES - 1)),
            ),
            criterion=UniformOffLabelsBCEWithLogitsLoss(),
            lr_scheduler=DEFAULT_LR_SCHEDULE
            if not is_big_model(name)
            else LONG_WARM_SCHEDULE,
            test_criterion=BinaryCrossEntropyLoss(),
            optimizer=OptimizerFactory(
                "Adam",
                lr=DEFAULT_LR,
            ),
        )
    )
    for name in SIMPLE_VIT_ARCHS
}


# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(baseline)
CONFIGS.update(bcos)

if __name__ == "__main__":
    configs_cli(CONFIGS)
