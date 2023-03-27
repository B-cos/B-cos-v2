import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from ..exceptions import AmbiguityError, EMANotFound, MetricsNotFoundError
from ..structure_constants import CHECKPOINT_LAST_FILENAME
from .metric_utils import Metrics

__all__ = [
    # main
    "get_state_dict_and_training_ckpt_from_save_dir",
    "ReloadTypes",
    # minor
    "change_state_dict_keys",
    "device_safe_load_state_dict_from_path",
    # extra utils
    "get_last_checkpoint_state_dict_from_save_dir",
    "get_last_checkpoint_path_in_save_dir",
]

# type aliases
PathLike = Union[str, Path]
StateDictType = Dict[str, Any]


class ReloadTypes:
    """
    Types of reloads.
    """

    BEST = "best"
    BEST_ANY = "best_any"
    LAST = "last"
    EPOCH = "epoch_"  # important: only use this as a prefix!

    @classmethod
    def validate(cls, value: str) -> bool:
        """
        Validate the given reload type (str).
        """
        return value in [
            cls.BEST,
            cls.BEST_ANY,
            cls.LAST,
        ] or value.startswith(cls.EPOCH)


def load_training_checkpoint_for_epoch_in(
    save_dir: PathLike,
    epoch: int,
) -> StateDictType:
    """
    Load the training checkpoint for the given epoch.

    Parameters
    ----------
    save_dir : PathLike
        The directory to load the checkpoint from.
    epoch : int
        The epoch to load the checkpoint for.

    Returns
    -------
    StateDictType
        The loaded state dict.
    """
    try:
        save_path = next(Path(save_dir).glob(f"epoch={epoch}-*.ckpt"))
    except StopIteration:
        raise FileNotFoundError(
            f"Tried loading checkpoint for epoch {epoch} but none was found in {save_dir}!"
        )
    return device_safe_load_state_dict_from_path(save_path)


class PLCheckpointStateDictLoader:
    """For loading state dicts from PyTorch Lightning checkpoints."""

    MODEL_STATE_DICT_PREFIX = "model."
    MODEL_STATE_DICT_EMA_PREFIX = "ema.module."

    @classmethod
    def get_model_state_dict(
        cls,
        training_ckpt: StateDictType,
        ema: bool = False,
    ) -> StateDictType:
        model_state_dict = training_ckpt["state_dict"]
        prefix = cls.MODEL_STATE_DICT_EMA_PREFIX if ema else cls.MODEL_STATE_DICT_PREFIX

        if ema and not all(k.startswith(prefix) for k in model_state_dict.keys()):
            raise EMANotFound("EMA state dict not found in training checkpoint!")

        model_state_dict = change_state_dict_keys(
            model_state_dict, prefix_filter=prefix
        )
        return model_state_dict

    @staticmethod
    def is_suitable_state_dict(training_ckpt: StateDictType) -> bool:
        return (
            "state_dict" in training_ckpt
            and "epoch" in training_ckpt
            and "pytorch-lightning_version" in training_ckpt
        )


class SimpleCheckpointStateDictLoader:
    """For loading state dicts from simple training checkpoints (non-PL)."""

    MODEL_STATE_DICT_KEY = "model_state_dict"

    @classmethod
    def get_model_state_dict(
        cls,
        training_ckpt: StateDictType,
        **kwargs: Any,
    ) -> StateDictType:
        if kwargs:
            warnings.warn(
                f"Got unexpected kwargs: {kwargs}. Ignoring "
                f"them as they are not used in this loader."
            )
        model_state_dict = training_ckpt[cls.MODEL_STATE_DICT_KEY]
        return model_state_dict

    @classmethod
    def is_suitable_state_dict(cls, training_ckpt: StateDictType) -> bool:
        return cls.MODEL_STATE_DICT_KEY in training_ckpt


LOADERS = [
    PLCheckpointStateDictLoader,
    SimpleCheckpointStateDictLoader,
]


def load_model_state_dict_from_training_ckpt(
    training_ckpt: StateDictType,
    ema: bool = False,
) -> StateDictType:
    """
    Load the model state dict from the given training checkpoint.
    This additional indirection allows to support non-PL checkpoints,
    if required in the future.

    Parameters
    ----------
    training_ckpt : StateDictType
        The training checkpoint to load the model state dict from.
    ema : bool
        Whether to load the EMA version of the model state dict.

    Returns
    -------
    StateDictType
        The loaded model state dict.
    """
    num_suitable_loaders = sum(
        loader.is_suitable_state_dict(training_ckpt) for loader in LOADERS
    )
    if num_suitable_loaders > 1:
        raise AmbiguityError("Multiple loaders are suitable for this checkpoint!")
    elif num_suitable_loaders == 0:
        raise NotImplementedError("Unsupported checkpoint format!")

    model_state_dict = None
    for loader in LOADERS:
        if loader.is_suitable_state_dict(training_ckpt):
            model_state_dict = loader.get_model_state_dict(training_ckpt, ema=ema)
            break
    assert model_state_dict is not None, "Should have found a suitable loader!"

    return model_state_dict


def get_state_dict_and_training_ckpt_from_save_dir(
    save_dir: PathLike,
    reload: str = "last",
    ema: bool = False,
    verbose: bool = False,
) -> Tuple[StateDictType, StateDictType]:
    """
    Get the state dict of the model.

    Parameters
    ----------
    save_dir : PathLike
        The directory to load the weights/state dict from.
    reload : str
        What weights to reload. Either "last", "best", "best_any" or "epoch_<N>".
        "best_any": will check both non-ema or ema (if they exist) weights for best accuracy
        "best" will load best ema weights if `ema=True` and best non-ema otherwise
        Default: "last"
    ema : bool
        Only consider EMA weights when reloading. Not effective if `reload` is "best_any".
    verbose : bool
        Print information about loaded epoch and top-1 accuracy.
        Default: False

    Returns
    -------
    Tuple[StateDictType, StateDictType]
        A tuple of the model state dict and the training checkpoint state dict.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Directory '{save_dir}' does not exist!")
    if not save_dir.is_dir():
        raise ValueError(f"'{save_dir}' is not a directory!")
    if not ReloadTypes.validate(reload):
        raise ValueError(f"Unknown reload type: '{reload}'")

    # get the last checkpoint
    training_ckpt = get_last_checkpoint_state_dict_from_save_dir(save_dir)

    # depending on the reload get the specific checkpoint
    if reload != ReloadTypes.LAST:
        if reload == ReloadTypes.BEST or reload == ReloadTypes.BEST_ANY:
            epoch, ema = _determine_best_epoch_and_ema_status(save_dir, ema, reload)
        elif reload.startswith(ReloadTypes.EPOCH):
            # parse provided epoch
            epoch = int(reload.split("_")[1])
        else:
            raise ValueError(f"Unknown reload type: '{reload}'")

        training_ckpt = load_training_checkpoint_for_epoch_in(save_dir, epoch)

    # get the model state dict
    state_dict = load_model_state_dict_from_training_ckpt(training_ckpt)

    # print extra information if need be
    if verbose:
        _print_information_of_loaded_state_dict(save_dir, ema, training_ckpt)

    return state_dict, training_ckpt


def _print_information_of_loaded_state_dict(
    save_dir: PathLike, ema: bool, training_ckpt: StateDictType
) -> None:
    """
    Print information about the loaded state dict.

    Parameters
    ----------
    save_dir : PathLike
        The directory containing training checkpoints.
    ema : bool
        Whether the EMA version of the state dict was loaded.
    training_ckpt : StateDictType
        The training checkpoint from which the state dict was loaded.
    """
    epoch = training_ckpt["epoch"]
    ema_suffix = "(EMA)" if ema else ""
    print(f"Loaded epoch: {epoch} {ema_suffix}")
    try:
        metrics = Metrics.from_experiment_dir(save_dir)
        val_acc = None
        metric_key = "eval_acc1" if not ema else "eval_acc1_ema"
        # for loop in case there are multiple same numbered epochs
        for epoch_i, val_acc_i in metrics[metric_key]:
            if epoch_i == epoch:
                val_acc = val_acc_i
                break
        assert val_acc is not None, "Unable to find val acc!"
        print(f"With validation accuracy: {val_acc:.2%}")
    except MetricsNotFoundError:
        print("No validation accuracy metrics found in checkpoint!")


def _determine_best_epoch_and_ema_status(
    save_dir: PathLike,
    ema: bool,
    reload: str,
) -> Tuple[int, bool]:
    """
    Determine the best epoch and whether to use EMA weights.
    Note: EMA is only considered if `ema=True`.

    Parameters
    ----------
    save_dir : PathLike
        The save directory containing the training checkpoints.
    ema : bool
        Only consider EMA weights if `ema=True`. Not effective if `reload="best_any"`.
    reload : str
        The reload type.
        "best": will load best ema weights if `ema=True` and best non-ema otherwise
        "best_any": will check both non-ema or ema (if they exist) weights for best accuracy

    Returns
    -------
    Tuple[int, bool]
        The best epoch and whether to use EMA weights.
    """
    try:
        metrics = Metrics.from_experiment_dir(save_dir)
    except MetricsNotFoundError:
        raise MetricsNotFoundError(  # change error message
            "Unable to find metrics! These are required to find the best checkpoint!"
        )

    if reload == ReloadTypes.BEST:
        if ema:
            best_epoch, best_acc = metrics.get_best_epoch_and_accuracy_ema()
        else:
            best_epoch, best_acc = metrics.get_best_epoch_and_accuracy()
    elif reload == ReloadTypes.BEST_ANY:
        best_epoch, best_acc = metrics.get_best_epoch_and_accuracy()
        try:
            best_epoch_ema, best_acc_ema = metrics.get_best_epoch_and_accuracy_ema()
        except EMANotFound:
            best_epoch_ema, best_acc_ema = -1, -1
        if best_acc_ema > best_acc:
            best_epoch, best_acc = best_epoch_ema, best_acc_ema
            ema = True
        else:
            ema = False
    else:
        raise ValueError(
            f"Unknown reload type: '{reload}'. Only 'best' or 'best_any' supported!"
        )

    return best_epoch, ema


def change_state_dict_keys(
    state_dict: StateDictType,
    prefix_filter: Optional[str] = None,
    filter_fn: Optional[Callable[[str], bool]] = None,
    remove_prefix: bool = True,
) -> StateDictType:
    """
    Change the keys of a state dict.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        The dict containing the state.
    prefix_filter: Optional[str]
        Optional prefix to filter the state dict by. Default: None
    filter_fn : Optional[Callable[[str], bool]]
        Optional function to filter the state dict by. Default: None
    remove_prefix : bool
        Whether to remove the prefix from the state dict's keys.
        Only works if `prefix_filter` is set. Default: True.

    Returns
    -------
    StateDictType
        The state dict with the changed keys.
    """
    # make sure both filters are not set
    if prefix_filter is not None and filter_fn is not None:
        raise ValueError("Only one of `prefix_filter` and `filter_fn` can be set.")

    # create filter function
    def _filter_fn(key: str) -> bool:
        if prefix_filter is not None:
            return key.startswith(prefix_filter)
        elif filter_fn is not None:
            return filter_fn(key)
        else:
            return True

    # optionally remove prefix
    def _key_mutator(key: str) -> str:
        if remove_prefix and prefix_filter is not None:
            return key[len(prefix_filter) :]
        else:
            return key

    # filter state dict
    state_dict = {
        _key_mutator(key): value for key, value in state_dict.items() if _filter_fn(key)
    }

    if len(state_dict) == 0 and (prefix_filter is not None or filter_fn is not None):
        warnings.warn("Filtering possibly resulted in an empty model state dict!")
    return state_dict


def device_safe_load_state_dict_from_path(path: PathLike) -> StateDictType:
    """
    Load state dict from path, but make sure it's on the CPU.

    Parameters
    ----------
    path : PathLike
        The path to load the state dict from.

    Returns
    -------
    StateDictType
        The loaded state dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tried loading last checkpoint but none found at: '{path}'"
        )

    state_dict = torch.load(path, map_location="cpu")

    return state_dict


def get_last_checkpoint_path_in_save_dir(save_dir: PathLike) -> Path:
    """
    Get the path of the last checkpoint in the given directory.
    """
    return Path(save_dir) / CHECKPOINT_LAST_FILENAME


def get_last_checkpoint_state_dict_from_save_dir(save_dir: PathLike) -> StateDictType:
    """
    Try to load the last checkpoint state dict.
    """
    save_path = get_last_checkpoint_path_in_save_dir(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(
            f"Tried loading last checkpoint but none found at: '{save_path}'"
        )
    return device_safe_load_state_dict_from_path(save_path)
