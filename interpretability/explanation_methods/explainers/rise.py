#
#     Adapted from https://github.com/eclique/RISE/blob/master/explanations.py
#
import functools
import os

import numpy as np
import torch
from skimage.transform import resize
from torch import nn as nn
from tqdm import tqdm

from interpretability.explanation_methods.utils import ExplainerBase, limit_n_images


class RISE(ExplainerBase, nn.Module):
    PATH_TEMPLATE = "target/rise_masks/masks-n{n}-s{s}-H{H}.npy"

    def __init__(self, model, batch_size=2, n=6000, s=6, p1=0.1, **kwargs):
        nn.Module.__init__(self)
        ExplainerBase.__init__(self, model)
        self._model_to_probabilties = getattr(
            model, "to_probabilities", functools.partial(torch.softmax, dim=1)
        )
        print("Using to_probabilities: ", self._model_to_probabilties)
        self.batch_size = batch_size
        print("Using internal batch size: ", self.batch_size)
        self.max_imgs_bs = 1  # external
        self.N = n
        self.s = s
        self.p1 = p1
        self.masks = None

    def generate_masks(self, savepath="masks.npy", input_size=None):
        np.random.seed(42)
        print("Generating masks for", input_size)
        p1, s = self.p1, self.s
        if not os.path.isdir(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(self.N, s, s) < p1
        grid = grid.astype("float32")

        masks = np.empty((self.N, *input_size))

        for i in tqdm(range(self.N)):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(
                grid[i], up_size, order=1, mode="reflect", anti_aliasing=False
            )[x : x + input_size[0], y : y + input_size[1]]
        masks = masks.reshape(-1, 1, *input_size)
        np.save(savepath, masks)

    def load_masks(self, filepath):
        if not os.path.exists(filepath):
            size = int(filepath[filepath.rfind("-H") + 2 : -len(".npy")])
            self.generate_masks(savepath=filepath[:-4], input_size=(size, size))
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]
        return self.masks

    @limit_n_images
    @torch.no_grad()
    # based on https://github.com/sukrutrao/Attribution-Evaluation (commit: 15f91181)
    def attribute(self, x, target, return_all=False):
        N = self.N
        s = self.s
        _, _, H, W = x.size()
        if self.masks is None or self.masks.shape[-1] != H:
            self.masks = self.load_masks(self.PATH_TEMPLATE.format(H=int(H), n=N, s=s))

        # Apply array of filters to the image
        stack = self.masks.to(x.device, non_blocking=True) * x.detach()

        # Get the model predictions for the masked images
        p = []
        for i in range(0, N, self.batch_size):
            p.append(
                self._model_to_probabilties(
                    self.model(stack[i : min(N, i + self.batch_size)])
                ).to("cpu", non_blocking=True)
            )
        p = torch.concat(p)

        # Number of classes
        CL = p.size(1)
        sal = p.detach().transpose(0, 1) @ self.masks.view(N, -1)
        sal = sal.view((CL, 1, H, W))
        sal = sal / N / self.p1
        if return_all:
            return sal
        return sal[int(target[0])][None]

    def attribute_selection(self, x, tgts):
        return self.attribute(x, tgts, return_all=True)[tgts]
