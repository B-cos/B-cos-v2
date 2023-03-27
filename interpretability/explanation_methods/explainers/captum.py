import torch
import torch.nn.functional as F
from captum.attr import (
    DeepLift,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    LayerGradCam,
    Saliency,
)

from interpretability.explanation_methods.utils import CaptumDerivative


class IntGrad(CaptumDerivative, IntegratedGradients):
    def __init__(self, model, n_steps=20, internal_batch_size=1):
        CaptumDerivative.__init__(
            self, model, n_steps=n_steps, internal_batch_size=internal_batch_size
        )
        IntegratedGradients.__init__(self, self.model)


class GB(CaptumDerivative, GuidedBackprop):
    def __init__(self, model):
        CaptumDerivative.__init__(self, model)
        GuidedBackprop.__init__(self, self.model)


class IxG(CaptumDerivative, InputXGradient):
    def __init__(self, model):
        CaptumDerivative.__init__(self, model)
        InputXGradient.__init__(self, self.model)


class Grad(CaptumDerivative, Saliency):
    def __init__(self, model):
        CaptumDerivative.__init__(self, model)
        self.configs.update({"abs": False})
        Saliency.__init__(self, self.model)


class GradCam(CaptumDerivative):
    def __init__(self, model, add_inverse=True, interpolate_mode="nearest"):
        CaptumDerivative.__init__(self, model)
        self.configs.update(
            {
                "relu_attributions": True,
                "interpolate_mode": interpolate_mode,
                "add_inverse": add_inverse,
            }
        )  # As in original GradCam paper
        # NOTE: we are assuming classifier comes before global average pooling
        try:
            self.features = model.get_feature_extractor()
            self.classifier = model.get_classifier()
        except AttributeError as e:
            raise AttributeError(
                "Model must implement get_feature_extractor and get_classifier methods"
            ) from e

    def attribute(self, img, target, **kwargs):
        with torch.no_grad():
            features = self.features(img)
        var_features = features.requires_grad_()
        out = F.adaptive_avg_pool2d(self.classifier(var_features), 1)[..., 0, 0]
        out[0, target].backward(inputs=[var_features])
        att = (var_features.grad.sum(dim=(-2, -1), keepdim=True) * var_features).sum(
            1, keepdim=True
        )
        if self.configs["relu_attributions"]:
            att.relu_()
        return LayerGradCam.interpolate(
            att, img.shape[-2:], interpolate_mode=self.configs["interpolate_mode"]
        )


class DeepLIFT(CaptumDerivative, DeepLift):
    def __init__(self, model):
        CaptumDerivative.__init__(self, model)
        DeepLift.__init__(self, self.model)
