import math  # noqa

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import configs_cli, update_config
from bcos.modules import norms
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

NUM_CLASSES = 1_000

# These are mainly based on the recipes from
# https://github.com/pytorch/vision/blob/93723b481d1f6e/references/classification/README.md
DEFAULT_BATCH_SIZE = 128  # * 8 GPUs = 1024 effective
DEFAULT_NUM_EPOCHS = 600
DEFAULT_LR = 1e-3
DEFAULT_CROP_SIZE = 176

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear

DEFAULT_OPTIMIZER = OptimizerFactory(name="Adam", lr=DEFAULT_LR)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
    warmup_method="linear",
    warmup_epochs=5,
    warmup_decay=0.01,
)

DEFAULTS = dict(
    data=dict(
        train_transform=ImageNetClassificationPresetTrain(
            crop_size=DEFAULT_CROP_SIZE,
            auto_augment_policy="ta_wide",
            random_erase_prob=0.1,
            is_bcos=True,
        ),
        test_transform=ImageNetClassificationPresetEval(
            crop_size=224,
            resize_size=232,
            is_bcos=True,
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        ra_repetitions=4,
    ),
    model=dict(
        is_bcos=True,
        # "name": None,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(
            b=2,
            max_out=1,
        ),
    ),
    criterion=UniformOffLabelsBCEWithLogitsLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    ema=dict(
        steps=32,
        decay=0.99998,
    ),
    use_agc=True,
)


# helper
def update_default(new_config):
    return update_config(DEFAULTS, new_config)


RESNET_DEPTHS = [50, 152]
resnets = {
    f"resnet_{depth}": update_default(
        dict(
            model=dict(
                name=f"resnet{depth}",
                args=dict(
                    stochastic_depth_prob=0.1,
                ),
            ),
        )
    )
    for depth in RESNET_DEPTHS
}

# the densenet is already a very small model (wrt #params),
# so regularization would probably only hurt performance
DENSENET_DEPTHS = [121]
densenets = {
    f"densenet_{depth}": update_default(
        dict(
            data=dict(
                ra_repetitions=None,
            ),
            model=dict(
                name=f"densenet{depth}",
            ),
        )
    )
    for depth in DENSENET_DEPTHS
}

convnext_bnu = {
    f"convnext_{size}_bnu": update_default(
        dict(
            model=dict(
                name=f"convnext_{size}",
                # in the default config, the norm layer is set to BNU
                # stochastic depth activated by default
            )
        )
    )
    for size in ["tiny", "base"]
}

convnext_pn = {
    f"convnext_{size}_pn": update_default(
        dict(
            model=dict(
                name=f"convnext_{size}",
                args=dict(
                    norm_layer=None,  # use model default pos norm
                ),
            )
        )
    )
    for size in ["tiny", "base"]
}


def change_configs_reduce_batch_size(configs, *names):
    for name in names:
        config = configs[name]
        config["trainer"]["accumulate_grad_batches"] = 2
        config["data"]["batch_size"] //= 2


# -------------------------------------------------------------------------

CONFIGS = dict()
CONFIGS.update(resnets)
CONFIGS.update(densenets)
CONFIGS.update(convnext_bnu)
CONFIGS.update(convnext_pn)

change_configs_reduce_batch_size(
    CONFIGS, "convnext_base_pn", "convnext_base_bnu", "resnet_152"
)


if __name__ == "__main__":
    configs_cli(CONFIGS)
