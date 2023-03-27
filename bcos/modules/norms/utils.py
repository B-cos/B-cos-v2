"""
Some extra utilities.
"""
from functools import wraps

__all__ = ["NoBias", "Unaffine"]


def _append_to_name(mod, suffix):
    old_name = mod.__class__.__name__

    def new_name():
        return old_name + suffix

    mod._get_name = new_name


def NoBias(make_layer):
    """
    Wraps around the layer making function and removes bias by setting it to
    None after instantiation.

    Parameters
    ----------
    make_layer
        The layer making function.

    Returns
    -------
    Modified layer making function which sets bias to None.
    """

    @wraps(make_layer)
    def init(*args, **kwargs):
        norm = make_layer(*args, **kwargs)

        assert (
            norm.bias is not None
        ), "It makes no sense to use this wrapper if you set affine=False!"
        norm.bias = None

        _append_to_name(norm, "NoBias")

        return norm

    if hasattr(make_layer, "__name__"):
        init.__name__ = make_layer.__name__ + "NoBias"
    if hasattr(make_layer, "__qualname__"):
        init.__qualname__ = make_layer.__qualname__ + "NoBias"

    return init


def Unaffine(make_layer):
    """
    Wraps around the layer making function and removes bias and weight
    by setting them to None after instantiation.

    Parameters
    ----------
    make_layer
        The layer making function.

    Returns
    -------
    Modified layer making function which sets bias and weight to None.
    """

    @wraps(make_layer)
    def init(*args, **kwargs):
        norm = make_layer(*args, **kwargs)

        assert (
            norm.bias is not None
        ), "It makes no sense to use this wrapper if you set affine=False!"
        norm.bias = None
        norm.weight = None

        _append_to_name(norm, "Unaffine")

        return norm

    if hasattr(make_layer, "__name__"):
        init.__name__ = make_layer.__name__ + "Unaffine"
    if hasattr(make_layer, "__qualname__"):
        init.__qualname__ = make_layer.__qualname__ + "Unaffine"

    return init
