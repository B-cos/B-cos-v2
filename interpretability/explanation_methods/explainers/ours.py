import torch

from interpretability.explanation_methods.utils import ExplainerBase

__all__ = ["Ours", "OursRelative"]


def Ours(model):
    assert hasattr(
        model,
        "attribute_selection",
    ), "model requires a 'attribute_selection' attribute for our explanation method!"
    return model


class OursRelative(ExplainerBase):
    """Generates mean-corrected explanations."""

    def __init__(self, model):
        assert hasattr(
            model, "explanation_mode"
        ), "model requires a 'explanation_mode' attribute for our (relative) explanation method!"
        super().__init__(model)
        from interpretability.explanation_methods.explainers.captum import IxG

        self.explainer = IxG(self.model_forward_with_mean_subtracted)

    def model_forward_with_mean_subtracted(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert (
            out.dim() == 2
        ), f"model output must be 2D (batch_size, num_classes) but is {out.ndim}D"
        return out - out.mean(dim=1, keepdim=True)

    def attribute(self, image, target, **kwargs):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479bd/training/trainer_base.py#L447-L464
        This method returns the contribution map according to Input x Gradient.
        Specifically, if the prediction model is a dynamic linear network, it returns the contribution map according
        to the linear mapping (IxG with detached dynamic weights).
        Args:
            image: Input image.
            target: Target class to check contributions for.
            kwargs: just for compatibility...
        Returns: Contributions for desired level.

        """
        _ = kwargs

        with self.model.explanation_mode():
            attribution_f = self.explainer
            att = attribution_f.attribute(image, target)

        return att

    def attribute_selection(self, image, targets, **kwargs):
        """
        From https://github.com/moboehle/B-cos/blob/4cd3b8ffc24d64c8b5b3262479/training/trainer_base.py#L467-L481
        Runs trainer.attribute for the list of targets.


        Args:
            image: Input image.
            targets: Target classes to check contributions for.
            kwargs: just for compatibility...

        Returns: Contributions for desired level.

        """
        _ = kwargs
        return torch.cat([self.attribute(image, t) for t in targets], dim=0)
