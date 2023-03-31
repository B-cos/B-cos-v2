"""
This module contains the main public API of the bcos package.
"""
# This module contains the preset transforms for ImageNet and C10
import bcos.data.presets as presets

# This contains single module level transforms. Among others, the AddInverse transform
# used to add inverse RGB channels to the input image.
import bcos.data.transforms as transforms

# This module contains the main models and pretrained models
import bcos.models as models

# This module contains the pretrained model entrypoints
import bcos.models.pretrained as pretrained

# This module contains the main nn.Modules of the package to build custom B-cos models
import bcos.modules as modules

# This module contains some factories to build optimizers and lr schedulers
import bcos.optim as optim

# This module contains the settings of the package
import bcos.settings as settings

# This module contains some utility functions/classes related to B-cos
from bcos.common import (
    BcosUtilMixin,
    explanation_mode,
    gradient_to_image,
    plot_contribution_map,
)
from bcos.version import __version__

__all__ = [
    "presets",
    "transforms",
    "models",
    "pretrained",
    "modules",
    "optim",
    "settings",
    "BcosUtilMixin",
    "explanation_mode",
    "gradient_to_image",
    "plot_contribution_map",
]
