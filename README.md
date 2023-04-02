# B-cos Networks v2
M. Böhle, N. Singh, M. Fritz, B. Schiele.

Improved B-cos Networks.

# Table of Contents
- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [`bcos` Package](#bcos-package)
  - [Training Environment Setup](#training-environment-setup)
    - [Setting Data Paths](#setting-data-paths)
- [Usage](#usage)
  - [Evaluation](#evaluation)
  - [Training](#training)
- [Model Zoo](#model-zoo)
- [License](#license)

# Introduction
This repository contains the code for the B-cos v2 models. (_Paper coming soon!_)

These models are more efficient and easier to train than the original B-cos models. Furthermore, we make a 
large number of pretrained B-cos models available for use.

If you want to take a quick look at the explanations the models generate, 
you can try out the Gradio web demo on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nps1ngh/B-cos).

If you prefer a more hands-on approach, 
you can take a look at the [demo notebook on `Colab`](https://colab.research.google.com/drive/1bdc1zdIVvv7XUJj8B8Toe6VMPYAsIT9w?usp=sharing)
or load the models directly via `torch.hub` as explained below.

If you simply want to copy the model definitions, we provide a minimal, single-file reference implementation including
explanation mode in [`extra/minimal_bcos_resnet.py`](extra/minimal_bcos_resnet.py)!


# Quick Start
You only need to make sure you have `torch` and `torchvision` installed. 

Then, loading the models via `torch.hub` is as easy as:

```python
import torch

# list all available models
torch.hub.list('B-cos/B-cos-v2')

# load a pretrained model
model = torch.hub.load('B-cos/B-cos-v2', 'resnet50', pretrained=True)
```

Inference and explanation visualization is as simple as:
```python
from PIL import Image
import matplotlib.pyplot as plt

# load image
img = model.transform(Image.open('cat.jpg'))
img = img[None].requires_grad_()

# predict and explain
model.eval()
expl_out = model.explain(img)
print("Prediction:", expl_out["prediction"])  # predicted class idx
plt.imshow(expl_out["explanation"])
plt.show()
```

Each of the models has its inference transform attached to it, accessible via `model.transform`.
Furthermore, each model has a `.explain()` method that takes an image tensor and returns a dictionary
containing the prediction and the explanation, and some extras.
<!-- TODO: add link to docs if end up making a site for them -->
See the [demo notebook](https://colab.research.google.com/drive/1bdc1zdIVvv7XUJj8B8Toe6VMPYAsIT9w?usp=sharing)
for more details on the `.explain()` method.

Furthermore, each model has a `get_classifier` and `get_feature_extractor` method that return the
classifier and feature extractor modules respectively. These can useful for fine-tuning the models!

<!-- =============================================================================================================== -->

# Installation
Depending on your use case, you can either install the `bcos` package 
or set up the development environment for training the models (for your custom models or for reproducing the results).

## `bcos` Package
If you are simply interested in using the models (pretrained or otherwise), 
then we provide a `bcos` package that can be installed via `pip`:

```bash
pip install bcos
```

This contains the models, their modules, transforms, and other utilities 
making it easy to use and build B-cos models.
Take a look at the public API [here](bcos/__init__.py). 
(I'll add a proper docs site if I have time or there's enough interest. 
Nonetheless, I have tried to keep the code well-documented, so it should be easy to follow.)

## Training Environment Setup
If you want to train your own B-cos models using this repository or are interested in reproducing the results, 
you can set up the development environment as follows:

Using `conda` (recommended, especially if you want to reproduce the results):
```bash
conda env create -f environment.yml
conda activate bcos
```

Using `pip`
```bash
pip install -r requirements-train.txt
```

### Setting Data Paths
You can either set the paths in [`bcos/settings.py`](bcos/settings.py) or set the environment variables
1. `DATA_ROOT`
2. `IMAGENET_PATH`

to the paths of the data directories.

The `DATA_ROOT` environment variable should point to the data root directory for CIFAR-10 
(will be automatically downloaded).
For ImageNet, the `IMAGENET_PATH` environment variable should point to the directory containing 
the `train` and `val` directories.



<!-- =============================================================================================================== -->

# Usage
For the `bcos` package, as mentioned earlier, take a look at the public API [here](bcos/__init__.py).

For evaluating or training the models, you can use the `evaluate.py` and `train.py` scripts, as follows:

## Evaluation
You can use evaluate the accuracy of the models on the ImageNet validation set using:
```bash
python evaluate.py --dataset ImageNet --hubconf resnet18
```
This will download the model from `torch.hub` and evaluate it on the ImageNet validation set.
The default batch size is 1, but you can change it using the `--batch-size` argument.
Replace `resnet18` with any of the other models listed in [Model Zoo](#model-zoo) that you wish to evaluate.

## Training
Short version:
```bash
python train.py \
  --dataset ImageNet \
  --base_network bcos_final \
  --experiment_name resnet18
```

Long version: See [TRAINING.md](TRAINING.md) for more details on 
how the setup works and how to train your own models.

# Model Zoo
Here are the ImageNet pre-trained models available in the model zoo.
You can find the links to the model weights below 
(uploaded to the [`Weights` GitHub release](https://github.com/B-cos/B-cos-v2/releases/tag/v0.0.1-weights)).

| Model/Entrypoint    | Top-1 Accuracy | Top-5 Accuracy | #Params | Download                                                                                                     |
|---------------------|----------------|----------------|---------|--------------------------------------------------------------------------------------------------------------|
| `resnet18`          | 68.736%        | 87.430%        | 11.69M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_18-68b4160fff.pth)          |
| `resnet34`          | 72.284%        | 90.052%        | 21.80M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_34-a63425a03e.pth)          |
| `resnet50`          | 75.882%        | 92.528%        | 25.52M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_50-ead259efe4.pth)          |
| `resnet101`         | 76.532%        | 92.538%        | 44.50M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_101-84c3658278.pth)         |
| `resnet152`         | 76.484%        | 92.398%        | 60.13M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_152-42051a77c1.pth)         |                            
| `resnext50_32x4d`   | 75.820%        | 91.810%        | 25.00M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnext_50_32x4d-57af241ab9.pth)   |
| `densenet121`       | 73.612%        | 91.106%        | 7.95M   | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_121-b8daf96afb.pth)       |
| `densenet161`       | 76.622%        | 92.554%        | 28.58M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_161-9e9ea51353.pth)       |
| `densenet169`       | 75.186%        | 91.786%        | 14.08M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_169-7037ee0604.pth)       |
| `densenet201`       | 75.480%        | 91.992%        | 19.91M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_201-00ac87066f.pth)       |
| `vgg11_bnu`         | 69.310%        | 88.388%        | 132.86M | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/vgg_11_bnu-34036029f0.pth)         |
|                     |                |                |         |                                                                                                              |
| `convnext_tiny`     | 77.488%        | 93.192%        | 28.54M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/convnext_tiny_pn-539b1bfb37.pth)   |
| `convnext_base`     | 79.650%        | 94.614%        | 88.47M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/convnext_base_pn-b0495852c6.pth)   |                                 
| `convnext_tiny_bnu` | 76.826%        | 93.090%        | 28.54M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/convnext_tiny_bnu-dbd7f5ef9d.pth)  |                                 
| `convnext_base_bnu` | 80.142%        | 94.834%        | 88.47M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/convnext_base_bnu-7c32a704b3.pth)  |
| `densenet121_long`  | 77.302%        | 93.234%        | 7.95M   | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/densenet_121_long-5175461597.pth)  |
| `resnet50_long`     | 79.468%        | 94.452%        | 25.52M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_50_long-ef38a88533.pth)     |
| `resnet152_long`    | 80.144%        | 94.116%        | 60.13M  | [link](https://github.com/B-cos/B-cos-v2/releases/download/v0.0.1-weights/resnet_152_long-0b4b434939.pth)    |

We'll add ViT models to the model zoo soon, stay tuned!
You can find these entrypoints in [`bcos/models/pretrained.py`](bcos/models/pretrained.py).



# License
This repository's code is licensed under the Apache License 2.0 
which you can find in the [LICENSE](./LICENSE) file.

The pre-trained models are trained on ImageNet (and are hence derived from it), which is 
licensed under the [ImageNet Terms of access](https://image-net.org/download),
which among others things, only allows non-commercial use of the dataset.
It is therefore your responsibility to check whether you have permission to use the 
pre-trained models for *your* use case.
