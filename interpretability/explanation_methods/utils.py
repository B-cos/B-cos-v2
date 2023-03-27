import numpy as np
import torch


class ExplainerImportFailedWarning(UserWarning):
    pass


def limit_n_images(func):
    """
    Function wrapper to assure that attributions for images are processed with a maximum batch size of self.max_imgs_bs.
    If max_imgs_bs is not set, this does nothing but calling the respective function.
    """

    def wrapper(mod, img, target, *args, **kwargs):
        if hasattr(mod, "max_imgs_bs") and len(img) > mod.max_imgs_bs:
            batch_size = mod.max_imgs_bs
            return torch.cat(
                [
                    func(
                        mod,
                        img[idx * batch_size : (idx + 1) * batch_size],
                        target[idx * batch_size : (idx + 1) * batch_size],
                        *args,
                        **kwargs
                    )
                    for idx in range(int(np.ceil(len(img) / batch_size)))
                ],
                dim=0,
            )

        return func(mod, img, target, *args, **kwargs)

    return wrapper


class ExplainerBase:

    """
    For consistency, every explanation method is derived from this explainer base and should adhere to the
    methods defined here. This allows for evaluating all explanations under the same pipeline.
    """

    def __init__(self, model):
        self.model = model

    def attribute(self, img, target, **kwargs):
        raise NotImplementedError("Need attribution method")

    def attribute_selection(self, img, tgts):
        raise NotImplementedError("Need attribution for selection of targets method")


class CaptumDerivative(ExplainerBase):

    """
    This class serves as a way to adapt all captum-based explanations methods to the same interface.
    """

    def __init__(self, model, **configs):
        """

        Args:
            trainer: Trainer class that holds the classification model as trainer.model
            **configs: configuration for the respective explanation method. Should be defined in explanation_configs.py.
        """
        super().__init__(model)
        self.configs = configs

    def attribute(self, img, target, **kwargs):
        """
        Calls the attribution method of the respective captum explanation method.
        """
        if "cuda" in kwargs and not kwargs["cuda"]:
            return self.__class__.__bases__[-1].attribute(
                self, img, target=torch.tensor(target), **self.configs
            )
        return self.__class__.__bases__[-1].attribute(
            self, img, target=torch.tensor(target).cuda(), **self.configs
        )

    def attribute_selection(self, img, targets):
        """
        Calls the attribution method for all targets in the list of targets
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        targets = np.array(targets, dtype=int).reshape(
            len(img), -1
        )  # Make sure it is a numpy array
        # Out is of size (bs, number of targets per image, in_channels, height, width)
        out = torch.zeros(*targets.shape[:2], *img.shape[1:]).type(torch.FloatTensor)
        for tgt_idx in range(targets.shape[1]):
            out[:, tgt_idx] = (
                self.attribute(img, target=(targets[:, tgt_idx]).tolist())
                .detach()
                .cpu()
            )
        return out.reshape(-1, *img.shape[1:]).cuda()
