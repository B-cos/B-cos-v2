# Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
  - [Training environment setup](#training-environment-setup)
    - [Setting data paths](#setting-data-paths)
- [Description of the training setup](#description-of-the-training-setup)
  - [Overview](#overview)
  - [Config System](#config-system)
  - [Training](#training)
- [Reproducing the results](#reproducing-the-results)
  - [CIFAR-10](#cifar-10)
  - [ImageNet](#imagenet)
- [Training your own models](#training-your-own-models)
  - [Adding a new dataset](#adding-a-new-dataset)
- [Using your trained models](#using-your-trained-models)

# Introduction
This document describes first how to set up the environment and data paths.
Then it explains how the training setup works, and how to train 
models for reproducing the results, or for training your own models.

# Setup
Just for completeness, we describe how to set up the training environment here.
You can skip this section if you have already set up the training environment as described in [README.md](README.md).

## Training environment setup
You can set up the development environment as follows:

Using `conda` (recommended, especially if you want to reproduce the results):
```bash
conda env create -f environment.yml
conda activate bcos
```

Using `pip`
```bash
pip install -r requirements-train.txt
```

### Setting data paths
You can either set the paths in [`bcos/settings.py`](bcos/settings.py) or set the environment variables
1. `DATA_ROOT`
2. `IMAGENET_PATH`
to the paths of the data directories.

The `DATA_ROOT` environment variable should point to the data root directory for CIFAR-10
(will be automatically downloaded).
For ImageNet, the `IMAGENET_PATH` environment variable should point to the directory containing
the `train` and `val` directories.


# Description of the training setup
## Overview
- Model definitions are in [`bcos/models`](bcos/models)
- Module definitions are in [`bcos/modules`](bcos/modules)
- [`bcos/common.py`](bcos/common.py) contains some common B-cos related utilities.
- Training configs are organized in [`bcos/experiments`](bcos/experiments).
  - It is organized as `bcos/experiments/<dataset>/<base_network>/`
  - `bcos/experiments/utils` contains some utility functions instead
- For training, 
  - we use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/),
  - the entry script is [`train.py`](train.py) or [`run_with_submitit.py`](run_with_submitit.py),
  - the trainer is in [`bcos/training/trainer.py`](bcos/training/trainer.py),
  - and some additional training utilities are in [`bcos/training/`](bcos/training/).
- Data and optimizer related stuff are in [`bcos/data`](bcos/data) and [`bcos/optim`](bcos/optim) respectively.

In the following, I'll describe the config system in more detail as 
it is crucial to understand how the training setup works.

## Config System
The config system is based on the [first B-cos version](https://github.com/moboehle/B-cos).
So if you're familiar with that, you can skip this.

The configs are stored in [`bcos/experiments`](bcos/experiments) and are organized as
`bcos/experiments/<dataset>/<base_network>/`, where `<dataset>` is either `CIFAR10` or `ImageNet`,
and `<base_network>` is the name of the collection of configs (this should have been renamed but 
hasn't due to legacy reasons, it would be more accurate to call it `config_collection` or something similar).

In each of the `<dataset>/<base_network>/` directories, there are the following files:
1. `experiment_parameters.py`
2. `model.py`

The `model.py` file contains a `get_model` function which takes as input the model config and returns the model,
doing any necessary processing of the config.

The `experiment_parameters.py` file contains all the configs in a dictionary called `CONFIGS`, which is a mapping
from config name (for a single _experiment_) to that experiment's config dictionary.
The config dictionary is a nested dictionary containing the data, model, optimizer, and other training 
related configurations.

See an example here: [`bcos/experiments/ImageNet/bcos_final/experiment_parameters.py`](https://github.com/B-cos/B-cos-v2/blob/main/bcos/experiments/ImageNet/bcos_final/experiment_parameters.py)

> **Warning**
> 
> The batch size in the config is the batch size per GPU and **NOT** the total effective batch size.
> We mention how many GPUs we used in the configs, and below in the [Reproducing the results](#reproducing-the-results) section.


## Training
As mentioned earlier, we use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training.
The trainer is in [`bcos/training/trainer.py`](bcos/training/trainer.py).
The trainer essentially consists of a general purpose [`LightningModule`](https://lightning.ai/docs/pytorch/stable/api/pytorch_lightning.core.LightningModule.html)
for training classification models called [`ClassificationLitModel`](bcos/training/trainer.py#L26).
The `ClassificationLitModel` class essentially wraps around some classification model (via our config system)
and implements the necessary methods for training via PyTorch Lightning.
For more details, see [`bcos/training/trainer.py`](bcos/training/trainer.py).

To run training, you can use [`train.py`](train.py) or [`run_with_submitit.py`](run_with_submitit.py).

The `train.py` script is for running training locally, and the `run_with_submitit.py` script is for running training
on a SLURM cluster using [`submitit`](https://github.com/facebookincubator/submitit), which has a nice API for
submitting jobs to SLURM and has additional benefits like automatic requeuing of unfinished jobs.
(The `run_with_submitit.py` script essentially just wraps the `train.py` script and adds some submitit related stuff.)

The following are the most relevant options for each:
- `train.py`
  - `--dataset`: The dataset to train on. Can be either `CIFAR10` or `ImageNet`.
  - `--base_network`: The base network to use. For CIFAR-10, this can be either `norm_ablations_final`
    For ImageNet, this can be either `bcos_final` or `bcos_final_long`. (You can add more obviously.)
  - `--experiment_name`: The name of the experiment to run. This should be one of the keys in the `CONFIGS` dictionary
    in the `experiment_parameters.py` file in the corresponding directory.
  - `--wandb_logger`: Whether to use [Weights & Biases](https://wandb.ai/site) for logging.
  - `--wandb_project`: The name of the W&B project to log to. (Required if `--wandb_logger` is set.)
  - `--distributed`: Whether to **FORCE** using DDP training. This is auto set otherwise.
- `run_with_submitit.py`
  - <all the options from `train.py`>
  - `--gpus`: The number of GPUs to use per node. (Required.)
  - `--nodes`: The number of nodes to use. (Required.)
  - `--partition`: The SLURM partition to use. (Required.)
  - `--timeout`: The timeout for the job in hours.
  - `--job_name`: The name of the job.

> **Warning**
> 
> As mentioned earlier (in the config system section), the batch size in the config 
> is the batch size per GPU and **NOT** the total effective batch size.
> This means you need the adjust the batch size in the config if your number of GPUs changes.


Look into the scripts for more details on their CLI options.
In the following, I'll describe the commands used for training the models.

# Reproducing the results
Note that a lot of factors like the PyTorch version, CUDA/cuDNN version, GPU model etc. can affect the results even 
after setting seeds.
Hence, depending upon your setup the results you get _may_ not be exactly the same as the ones reported.
Nonetheless, we tried to make the training reproducible
by using Lightning's [`pl.seed_everything`](https://pytorch-lightning.readthedocs.io/en/latest/common/seed.html).

The commands are below. These will create a respective experiment run folder in 
`experiments/<dataset>/<base_network>/<experiment_name>/` containing the logs, checkpoints, etc.

> **Note**
>
> This is **different** the `bcos/experiments` directory mentioned earlier 
(this is done, so that one can delete the artifacts from a failed run without deleting the config alongside with it).
The configs are stored in `bcos/experiments` and the experiment runs are stored in `experiments` 
(it's in the root).


## CIFAR-10

For CIFAR-10, we trained on a single GPU and used the following command (template):
```bash
python train.py \
    --dataset CIFAR10 \
    --base_network norm_ablations_final \
    --experiment_name <experiment_name>
```

> **Warning**
> 
> This is only specific to our CIFAR10 models:
> `experiment_name`s ending with `-nomaxout` have no MaxOut activation, **BUT** the ones without have `maxout=2` (like in v1).
>
> For ImageNet, all the models have ***no*** MaxOut activation.

## ImageNet
For ImageNet, we trained on **4** GPUs for the `bcos_final` configs and **8** GPUs for the `bcos_final_long` configs.
The commands (templates) are as follows:
```bash
python run_with_submitit.py \
    --dataset ImageNet \
    --base_network bcos_final \
    --experiment_name <experiment_name> \
    --gpus 4 \
    --nodes 1
```

```bash
python run_with_submitit.py \
    --dataset ImageNet \
    --base_network bcos_final_long \
    --experiment_name <experiment_name> \
    --gpus 4 \
    --nodes 2
```

> **Note**
> 
> 4 GPUs per node times 2 nodes = 8 GPUs!

# Training your own models
If you want to train your own models, you simply need to:
1. Add your model definition in [`bcos/models`](bcos/models).
2. Create a new directory in `bcos/experiments/<dataset>/` with the name of your config collection.
3. Create a `model.py` file in that directory and implement the `get_model` function.
4. Create a `experiment_parameters.py` file in that directory and add your configs to the `CONFIGS` dictionary.
   You can simply copy the `experiment_parameters.py` file from one of the existing config collections and modify it.
5. Run training as described above.


## Adding a new dataset
If you want to add a new dataset, you will need to add a new datamodule class 
(see Lightning's [`LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html))
in [`bcos/data/datamodules.py`](bcos/data/datamodules.py), which should inherit from 
[`ClassificationDataModule`](bcos/data/datamodules.py#L27) (otherwise it does not get registered).

You can look at the existing datamodule classes for reference.
Your datamodule class must have a name of the form `<dataset_name>DataModule` (e.g. `CIFAR10DataModule`).
Also, it must have a `NUM_CLASSES` class attribute which is the number of classes in the dataset.

Then you should create configs as described above in `bcos/experiments/<dataset_name>/` and run training as described 
above, where `<dataset_name>` is the name of your datamodule class minus the `DataModule` suffix.

If you're using custom data, then you can create a custom datamodule class 
using `torchvision`'s [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder).


# Using your trained models
To use your trained models, you can load them up using the `Experiment` class as follows:
```python
from bcos.experiments.utils import Experiment

exp = Experiment(
    "CIFAR10",
    "norm_ablations_final",
    "resnet_20_bnu-linear-nomaxout",
)

model = exp.load_trained_model(reload="last")  # reload="best" to load the best model
datamodule = exp.get_datamodule()
```

You can then use the `model` and `datamodule` objects to do inference, etc.
For more details, look at the [`Experiment` class](bcos/experiments/utils/experiment_utils/experiment_utils.py)
and its methods.

If you simply want to measure the accuracy of your model on a dataset, you can use the `evaluate.py` script:
```bash
python evaluate.py \
    --dataset CIFAR10 \
    --base_network norm_ablations_final \
    --experiment_name resnet_20_bnu-linear-nomaxout
    --reload last
```

If you want to run localisation analysis on the trained model, 
you can use the [`localisation.py`](interpretability/analyses/localisation.py) 
script:
```bash
python -m interpretability.analyses.localisation \
    --reload last \
    --analysis_config 500_3x3 \
    --explainer_name Ours \
    --smooth 1 \
    --batch_size 64 \
    --save_path "experiments/CIFAR10/norm_ablations_final/resnet_20_bnu-linear-nomaxout/"
```
> **Note**
> 
> Note that `--smooth 1` means no smoothing (as kernel size = 1)
and `--batch_size 64` may not be effective depending upon the explanation method used (forced to be 1).

This will create a folder `localisation_analysis` in the experiment run directory containing the results.

An example command for running localisation analysis on an ImageNet model is:
```bash
python -m interpretability.analyses.localisation \
    --reload best_any \
    --analysis_config 250_2x2 \
    --explainer_name Ours \
    --smooth 15 \
    --batch_size 64 \
    --save_path "experiments/ImageNet/bcos_final/resnet_18/"
```

---

Feel free to open an issue if you have any questions or problems! :)

<!-- TODO: Add link to fine-tuning notebook, once complete -->
