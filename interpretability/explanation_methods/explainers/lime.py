#
#     Adapted from https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
#
import copy
import functools

import numpy as np
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from interpretability.explanation_methods.utils import ExplainerBase, limit_n_images


class Lime(ExplainerBase, lime_image.LimeImageExplainer):
    def __init__(
        self,
        model,
        num_samples=256,
        num_features=3,
        kernel_size=1,
        batch_size=2,
        num_classes=1000,
    ):
        ExplainerBase.__init__(self, model)
        self._model_to_probabilties = getattr(
            model, "to_probabilities", functools.partial(torch.softmax, dim=1)
        )
        print("Using to_probabilities: ", self._model_to_probabilties)
        lime_image.LimeImageExplainer.__init__(self, verbose=False)
        self.segmenter = SegmentationAlgorithm(
            "quickshift", kernel_size=kernel_size, max_dist=200, ratio=0.2
        )
        self.max_imgs_bs = 1
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_features = num_features
        self.batch_size = batch_size
        print(f"Using internal batch size of {batch_size} for Lime.")
        self.device = next(model.parameters()).device
        self.is_bcos_model = hasattr(model, "explanation_mode")
        if self.is_bcos_model:
            print("B-cos model detected. Will patch to use 6 channel inputs")
        else:
            print("B-cos model not detected. Will use 3 channel inputs")

    def pred_f(self, input_samples):
        in_tensor = self.make_input_tensor(input_samples)
        out_tensor = self.model(in_tensor)
        out = self._model_to_probabilties(out_tensor)
        return out.detach().cpu().numpy()

    def make_input_tensor(self, img_list):
        result = (
            torch.stack([torch.from_numpy(t) for t in img_list], dim=0)
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True)
        )
        if self.is_bcos_model:
            result = torch.concat([result, 1 - result], dim=1)
        return result

    @limit_n_images
    @torch.no_grad()
    def attribute(self, img, target, return_all=False):
        explanation = self.explain_instance(
            img[0, :3].permute(1, 2, 0).detach().cpu().numpy(),
            self.pred_f,
            labels=range(self.num_classes),
            top_labels=None,
            num_samples=self.num_samples,
            segmentation_fn=self.segmenter,
            batch_size=self.batch_size,
        )

        if return_all:
            return torch.cat(
                [
                    torch.from_numpy(
                        np.array(
                            explanation.get_image_and_mask(
                                t,
                                hide_rest=True,
                                positive_only=True,
                                num_features=self.num_features,
                            )[1][None, None],
                            dtype=float,
                        )
                    )
                    for t in range(self.num_classes)
                ],
                dim=0,
            )

        return torch.from_numpy(
            np.array(
                explanation.get_image_and_mask(
                    int(np.array(target)),
                    hide_rest=True,
                    positive_only=True,
                    num_features=self.num_features,
                )[1][None, None],
                dtype=float,
            )
        )

    def attribute_selection(self, x, tgts):
        return self.attribute(x, tgts, return_all=True)[tgts]

    def data_labels(
        self, image, fudged_image, segments, classifier_fn, num_samples, batch_size=10
    ):
        """
        SAME AS BASE, just deleted tqdm.
        Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features).reshape(
            (num_samples, n_features)
        )
        labels = []
        data[0, :] = 1
        imgs = []
        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
