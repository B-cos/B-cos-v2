"""
This module contains utilities related to B-cos models.
None of this is "essential" to training or doing inference with the models.
(Most of the stuff can be done quickly and easily in a few lines of code.)
However, they are useful for e.g. visualizing the explanations etc.
So essentially it's a collection of convenience/helper functions/classes.
"""
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    # this isn't supposed to be a hard dependency
    import matplotlib
    import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if torch.__version__ < "2.0":
    from torch.autograd.grad_mode import _DecoratorContextManager  # noqa
else:
    from torch.utils._contextlib import _DecoratorContextManager  # noqa

__all__ = [
    "BcosUtilMixin",
    "explanation_mode",
    "gradient_to_image",
    "plot_contribution_map",
]


TensorLike = Union[Tensor, np.ndarray]


class BcosUtilMixin:
    """
    This mixin defines useful helpers for dealing with explanations.
    This is just a convenience to attach useful B-cos specific functionality.

    The parameters to ``__init__`` are just passed to the actual base class (e.g. ``torch.nn.Module``).

    Notes
    -----
    Since this is a mixin, if you want to use this, you need to inherit from this first
    and then from the actual base class (e.g. `torch.nn.Module`).


    Examples
    --------
    >>> from bcos.modules import BcosConv2d
    >>> class MyModel(BcosUtilMixin, torch.nn.Module):
    ...     def __init__(self, in_chan: int, out_chan: int):
    ...         super().__init__()
    ...         self.linear = BcosConv2d(in_chan, out_chan, 3)
    ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...         return self.linear(x)
    >>> model = MyModel(6, 16)
    >>> model.explain(torch.rand(1, 6, 32, 32))  # get explain method
    >>> with model.explanation_mode():  # explanation mode ctx (assuming we have detachable modules)
    ...     ...  # do something with explanation mode activated

    Parameters
    ----------
    args: Any
        Positional arguments to pass to the parent class.
    kwargs: Any
        Keyword arguments to pass to the parent class.
    """

    to_probabilities = torch.sigmoid
    """ Function to convert model outputs to probabilties. """

    def __init__(self, *args: Any, **kwargs: Any):
        self.__explanation_mode_ctx = explanation_mode(self)  # type: ignore
        super().__init__(*args, **kwargs)

    def explanation_mode(self) -> "explanation_mode":
        """
        Returns a context manager which puts model in explanation mode
        and when exiting puts it in normal mode back again.

        Returns
        -------
        explanation_mode
            The context manager which puts model in and out to/from explanation mode.
        """
        return self.__explanation_mode_ctx

    def explain(
        self,
        in_tensor,
        idx=None,
        **grad2img_kwargs,
    ) -> "Dict[str, Any]":
        """
        Generates an explanation for the given input tensor.
        This is not a generic explanation method, but rather a helper for simply getting explanations.
        It is intended for simple use cases (simple exploration, debugging, etc.).

        Parameters
        ----------
        in_tensor : Tensor
            The input tensor to explain. Must be 4-dimensional and have batch size of 1.
        idx : int, optional
            The index of the output to explain. If None, the prediction is explained.
        grad2img_kwargs : Any
            Additional keyword arguments passed to `gradient_to_image` method
            for generating the explanation.

        Examples
        --------
        Here is an example of how to use this method to generate and visualize an explanation and a contribution map:

        >>> model = ...  # instantiate some B-cos model
        >>> img = ...  # instantiate some input image tensor
        >>> expl_out = model.explain(img)
        >>> expl_out["prediction"]
        932

        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(expl_out["explanation"])
        >>> plt.show()  # show the explanation

        >>> model.plot_contribution_map(expl_out["contribution_map"])
        >>> plt.show()  # show the contribution map

        Warnings
        --------
        This method is NOT optimized for speed.
        Also, on a more general note: Care should be taken when generating explanations during training,
        as the gradients might be different from during inference.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the explanation and additional information.
            Namely, the following keys are present:
            - "prediction": The prediction of the model.
            - "explained_class_idx": The class (index) of the explained output.
            - "dynamic_linear_weights": The dynamic linear weights of the model (`in_tensor.grad`).
            - "contribution_map": The contribution map of the model prediction.
            - "explanation": The explanation of the model prediction.
        """
        if in_tensor.ndim == 3:
            raise ValueError("Expected 4-dimensional input tensor")
        if in_tensor.shape[0] != 1:
            raise ValueError("Expected batch size of 1")
        if not in_tensor.requires_grad:
            warnings.warn(
                "Input tensor did not require grad! Has been set automatically to True!"
            )
            in_tensor.requires_grad = True  # nonsense otherwise
        if self.training:  # noqa
            warnings.warn(
                "Model is in training mode! "
                "This might lead to unexpected results! Use model.eval()!"
            )

        result = dict()
        with torch.enable_grad(), self.explanation_mode():
            # fwd + prediction
            out = self(in_tensor)  # noqa
            pred_out = out.max(1)
            result["prediction"] = pred_out.indices.item()

            # select output (logit) to explain
            if idx is None:  # explain prediction
                to_be_explained_logit = pred_out.values
                result["explained_class_idx"] = pred_out.indices.item()
            else:  # user specified idx
                to_be_explained_logit = out[0, idx]
                result["explained_class_idx"] = idx

            to_be_explained_logit.backward(inputs=[in_tensor])

        # get weights and contribution map
        result["dynamic_linear_weights"] = in_tensor.grad
        result["contribution_map"] = (in_tensor * in_tensor.grad).sum(1)

        # generate (color) explanation
        result["explanation"] = gradient_to_image(
            in_tensor[0], in_tensor.grad[0], **grad2img_kwargs
        )

        return result

    @staticmethod  # to make it easier when using torch.hub
    def gradient_to_image(
        image: "Tensor",
        linear_mapping: "Tensor",
        smooth: int = 15,
        alpha_percentile: float = 99.5,
    ) -> "np.ndarray":
        """
        From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.
        Computing color image from dynamic linear mapping of B-cos models.

        Parameters
        ----------
        image: Tensor
            Original input image (encoded with 6 color channels)
            Shape: [C, H, W] with C=6
        linear_mapping: Tensor
            Linear mapping W_{1\rightarrow l} of the B-cos model
            Shape: [C, H, W] same as image
        smooth: int
            Kernel size for smoothing the alpha values
        alpha_percentile: float
            Cut-off percentile for the alpha value

        Returns
        -------
        np.ndarray
            image explanation of the B-cos model.
            Shape: [H, W, C] (C=4 ie RGBA)
        """
        return gradient_to_image(image, linear_mapping, smooth, alpha_percentile)

    @staticmethod  # to make it easier when using torch.hub
    def plot_contribution_map(
        contribution_map: TensorLike,
        ax: Optional["plt.Axes"] = None,
        vrange: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        hide_ticks: bool = True,
        cmap: str = "bwr",
        percentile: float = 99.5,
    ) -> "Tuple[plt.Axes, matplotlib.image.AxesImage]":
        """
        From https://github.com/moboehle/B-cos/blob/0023500cea7b/interpretability/utils.py#L78-L115
        For an example of how to use this, see docstring for `explain`.

        Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
        As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
        ranges from (-max(abs(contribution_map), max(abs(contribution_map)).

        Parameters
        ----------
        contribution_map: TensorLike
            (H, W) matrix of contributions to visualize.
        ax: Optional[plt.Axes]
            Axis on which to plot. If None, a new figure is created.
        vrange: Optional[float]
            If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
            If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
            overwritten by vmin or vmax.
        vmin: Optional[float]
            Manually overwrite the minimum value for the colormap range instead of using -vrange.
        vmax: Optional[float]
            Manually overwrite the maximum value for the colormap range instead of using vrange.
        hide_ticks: bool
            Sets the axis ticks to []
        cmap: str
            colormap to use for the contribution map plot.
        percentile: float
            If percentile is given, this will be used as a cut-off for the attribution maps.

        Returns
        -------
        ax: plt.Axes
            The axis on which the contribution map was plotted.
        im: matplotlib.image.AxesImage
            The image object of the contribution map.
        """
        return plot_contribution_map(
            contribution_map,
            ax,
            vrange,
            vmin,
            vmax,
            hide_ticks,
            cmap,
            percentile,
        )

    def attribute(
        self,
        image: Union[Tensor, Tuple[Tensor]],
        target: Union[int, Tuple[int], Tensor, List[int]],
        **kwargs: Any,
    ) -> Tensor:
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479bd/training/trainer_base.py#L447-L464
        This is essentially just a Captum-IxG-dependent contribution generator for B-cos models.

        This method returns the contribution map according to Input x Gradient.
        Specifically, if the prediction model is a dynamic linear network, it returns the contribution map according
        to the linear mapping (IxG with detached dynamic weights).

        Parameters
        ----------
        image: Tensor | Tuple[Tensor]
            Input image(s). Shape (1, C, H, W).
        target: int | tuple[int] | Tensor | list[int]
            Target class to check contributions for.
        kwargs: Any
            just for compatibility...

        Returns
        -------
        Tensor
            Contributions for desired level.
        """
        _ = kwargs
        # install captum if you want to use captum dependent contribution generation
        # this will fail otherwise
        from interpretability.explanation_methods.explainers.captum import IxG

        with self.explanation_mode():
            attribution_f = IxG(self)
            att = attribution_f.attribute(image, target)

        return att

    def attribute_selection(
        self,
        image: Tensor,
        targets: Union[Tuple[int], Tensor, List[int]],
        **kwargs: Any,
    ) -> Tensor:
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs `.attribute` for multiple targets and concatenates the results.

        Parameters
        ----------
        image: Tensor
            Input image.
        targets: tuple[int] | Tensor | list[int]
            Target classes to check contributions for.
        kwargs: Any
            just for compatibility...

        Returns
        -------
        Tensor
            Contributions for desired level.
        """
        _ = kwargs
        return torch.cat([self.attribute(image, t) for t in targets], dim=0)


class explanation_mode(_DecoratorContextManager):
    """
    A context manager which activates and puts model in to explanation
    mode and deactivates it afterwards.
    Can also be used as a decorator.

    Parameters
    ----------
    model : nn.Module
        The model to put in explanation mode.
    """

    def __init__(self, model: "nn.Module"):
        self.model = model
        self.expl_modules = None

    def find_expl_modules(self) -> None:
        """Finds all modules which have a `set_explanation_mode` method."""
        self.expl_modules = [
            m for m in self.model.modules() if hasattr(m, "set_explanation_mode")
        ]

    def __enter__(self):
        """
        Put model in explanation mode.
        """
        if self.expl_modules is None:
            self.find_expl_modules()

        for m in self.expl_modules:
            m.set_explanation_mode(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Turn off explanation mode for model.
        """
        for m in self.expl_modules:
            m.set_explanation_mode(False)


def gradient_to_image(image, linear_mapping, smooth=15, alpha_percentile=99.5):
    """
    From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.
    Computing color image from dynamic linear mapping of B-cos models.

    Parameters
    ----------
    image: Tensor
        Original input image (encoded with 6 color channels)
        Shape: [C, H, W] with C=6
    linear_mapping: Tensor
        Linear mapping W_{1\rightarrow l} of the B-cos model
        Shape: [C, H, W] same as image
    smooth: int
        Kernel size for smoothing the alpha values
    alpha_percentile: float
        Cut-off percentile for the alpha value. In range [0, 100].

    Returns
    -------
    np.ndarray
        image explanation of the B-cos model.
        Shape: [H, W, C] (C=4 ie RGBA)
    """
    # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
    contribs = (image * linear_mapping).sum(0, keepdim=True)  # [H, W]
    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )
    # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
    rgb_grad = rgb_grad.clamp(min=0)
    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)  # [3, H, W]

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    # Only show positive contributions
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)

    rgb_grad = torch.concatenate([rgb_grad, alpha], dim=0)  # [4, H, W]
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.permute(1, 2, 0)
    return grad_image.detach().cpu().numpy()


def plot_contribution_map(
    contribution_map: TensorLike,
    ax: Optional["plt.Axes"] = None,
    vrange: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hide_ticks: bool = True,
    cmap: str = "bwr",
    percentile: float = 99.5,
) -> "Tuple[plt.Axes, matplotlib.image.AxesImage]":
    """
    From https://github.com/moboehle/B-cos/blob/0023500cea7b/interpretability/utils.py#L78-L115
    For an example of how to use this, see docstring for `BcosMixin.explain`.

    Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
    As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
    ranges from (-max(abs(contribution_map), max(abs(contribution_map)).

    Parameters
    ----------
    contribution_map: TensorLike
        (H, W) matrix of contributions to visualize.
    ax: Optional[plt.Axes]
        Axis on which to plot. If None, a new figure is created.
    vrange: Optional[float]
        If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
        If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
        overwritten by vmin or vmax.
    vmin: Optional[float]
        Manually overwrite the minimum value for the colormap range instead of using -vrange.
    vmax: Optional[float]
        Manually overwrite the maximum value for the colormap range instead of using vrange.
    hide_ticks: bool
        Sets the axis ticks to []
    cmap: str
        colormap to use for the contribution map plot.
    percentile: float
        If percentile is given, this will be used as a cut-off for the attribution maps.

    Returns
    -------
    ax: plt.Axes
        The axis on which the contribution map was plotted.
    im: matplotlib.image.AxesImage
        The image object of the contribution map.
    """
    assert (
        contribution_map.ndim == 2
    ), "Contribution map is supposed to only have 2 spatial dimensions."
    if isinstance(contribution_map, torch.Tensor):
        contribution_map = contribution_map.detach().cpu().numpy()
    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)

    if ax is None:
        import matplotlib.pyplot as plt  # noqa

        fig, ax = plt.subplots(1)

    if vrange is None or vrange == "auto":
        vrange = np.max(np.abs(contribution_map.flatten()))
    im = ax.imshow(
        contribution_map,
        cmap=cmap,
        vmin=-vrange if vmin is None else vmin,
        vmax=vrange if vmax is None else vmax,
    )

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, im
