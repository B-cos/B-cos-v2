from torch import nn

from bcos.common import BcosUtilMixin

__all__ = ["DetachableModule", "BcosSequential"]


class DetachableModule(nn.Module):
    """
    A base module for modules which can detach dynamic weights from the graph,
    which is necessary to calculate explanations.
    """

    def __init__(self):
        super().__init__()
        self.detach = False

    def set_explanation_mode(self, activate: bool = True) -> None:
        """
        Turn explanation mode on or off.

        Parameters
        ----------
        activate : bool
            Turn it on.
        """
        self.detach = activate

    @property
    def is_in_explanation_mode(self) -> bool:
        """
        Whether the module is in explanation mode or not.
        """
        return self.detach


class BcosSequential(BcosUtilMixin, nn.Sequential):
    """
    Wrapper for models which are nn.Sequential at the "root" module level.
    This only adds helper functionality from `BcosMixIn`.
    """

    def __init__(self, *args):
        super().__init__(*args)
