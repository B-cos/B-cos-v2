import warnings

from interpretability.explanation_methods.utils import ExplainerImportFailedWarning

# set to `False` to get import errors for missing dependencies
WARN_INSTEAD_OF_RAISE = True


def warn_or_raise(import_error, explainer_tried_to_import):
    if WARN_INSTEAD_OF_RAISE:
        warnings.warn(
            f"Failure when trying to import: '{explainer_tried_to_import}'!\n"
            "The explainer will not be available! {"
            f"{import_error}",
            category=ExplainerImportFailedWarning,
        )
    else:
        raise import_error


# this allows to still run localisation for a specific explainer without installing all others!
try:
    from interpretability.explanation_methods.explainers.captum import (
        GB,
        DeepLIFT,
        Grad,
        GradCam,
        IntGrad,
        IxG,
    )

    # dependent on IxG
    from interpretability.explanation_methods.explainers.ours import Ours, OursRelative

    HAS_CAPTUM = True
except ImportError as ie:
    warn_or_raise(ie, "Captum")
    HAS_CAPTUM = False

try:
    from interpretability.explanation_methods.explainers.rise import RISE

    HAS_RISE = True
except ImportError as ie:
    warn_or_raise(ie, "RISE")
    HAS_RISE = False
try:
    from interpretability.explanation_methods.explainers.lime import Lime

    HAS_LIME = True
except ImportError as ie:
    warn_or_raise(ie, "Lime")
    HAS_LIME = False
try:
    from interpretability.explanation_methods.explainers.occlusion import Occlusion

    HAS_OCCL = True
except ImportError as ie:
    warn_or_raise(ie, "Occlusion")
    HAS_OCCL = False

from interpretability.explanation_methods.explanation_configs import explainer_configs

explainer_map = {}
"""Mapping from explainer name to explainer class."""

if HAS_OCCL:
    explainer_map["Occlusion"] = Occlusion
if HAS_RISE:
    explainer_map["RISE"] = RISE
if HAS_LIME:
    explainer_map["LIME"] = Lime
if HAS_CAPTUM:
    explainer_map.update(
        {
            "GCam": GradCam,
            "IntGrad": IntGrad,
            "GB": GB,
            "IxG": IxG,
            "Grad": Grad,
            "DeepLIFT": DeepLIFT,
            "Ours": Ours,
            "OursRelative": OursRelative,
        }
    )


def get_explainer(model, explainer_name, config_name, **config_overrides):
    try:
        explainer_config = explainer_configs[explainer_name][config_name]
        updated_config = {**explainer_config, **config_overrides}
        try:
            return explainer_map[explainer_name](model, **updated_config)
        except TypeError:  # in case of unexpected keyword arguments
            warnings.warn(
                f"Ignoring overrides {config_overrides} for explainer config!"
            )
            return explainer_map[explainer_name](model, **explainer_config)
    except KeyError:
        raise KeyError(
            f"Explainer '{explainer_name}' with config '{config_name}' not found! "
            "Make sure you have the necessary dependencies installed!"
            # set WARN_INSTEAD_OF_RAISE to `False` to raise an error instead of a warning
        )
