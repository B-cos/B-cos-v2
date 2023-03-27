"""
Utilities for loading models, configs, etc.
"""
import difflib
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from ..config_utils import get_configs_and_model_factory, update_config
from ..exceptions import ModelFactoryDoesNotSupportPretrainedError
from ..structure_constants import BASE_EXPERIMENTS_DIRECTORY, ROOT
from .loading_utils import (
    device_safe_load_state_dict_from_path,
    get_last_checkpoint_path_in_save_dir,
    get_state_dict_and_training_ckpt_from_save_dir,
)
from .metric_utils import Metrics

__all__ = [
    "Experiment",  # main API
]


class Experiment:
    """
    Utility class for loading models and configs.

    Examples
    --------
    >>> from bcos.experiments.utils import Experiment
    >>> exp = Experiment("ImageNet", "bcos_final", "resnet_50")
    >>> exp.save_dir                 # gets the save directory
    >>> exp.config                   # gets the config
    >>> exp.get_model()              # gets a new model instance
    >>> exp.available_checkpoints()  # gets the available checkpoints
    >>> exp.get_datamodule()         # gets a new datamodule instance (requires PL)
    >>> exp.load_metrics()           # loads the metrics
    >>> exp.load_trained_model()     # creates a model and loads the weights from training ckpt
    """

    def __init__(
        self,
        path_or_dataset: Union[str, Path],
        base_network: Optional[str] = None,
        experiment_name: Optional[str] = None,
        base_directory: str = BASE_EXPERIMENTS_DIRECTORY,
    ):
        """
        Initialize an Experiment object.

        Parameters
        ----------
        path_or_dataset : Union[str, Path]
            The path to a stored experiment OR the name of the dataset.
        base_network : Optional[str]
            The base network. Only provide this if `path_or_dataset` is a dataset.
        experiment_name : Optional[str]
            The experiment name. Only provide this if `path_or_dataset` is a dataset.
        base_directory : str
            The base directory. Default: "./experiments"
        """
        if (
            isinstance(path_or_dataset, Path) or "/" in path_or_dataset
        ):  # with / will assume it's a path
            assert (
                base_network is None and experiment_name is None
            ), f"Path provided ('{path_or_dataset=}' b/c it contains '/')! Other arguments should be None!"
            path = Path(path_or_dataset)
            experiment_name = path.name
            base_network = path.parent.name
            dataset = path.parent.parent.name
            base_directory = path.parent.parent.parent
        else:  # assume it's a dataset
            dataset = path_or_dataset
            assert (
                base_network is not None and experiment_name is not None
            ), "Provide other arguments too!"

        configs, get_model = get_configs_and_model_factory(dataset, base_network)

        self._get_model = get_model
        """ The model factory. """

        if experiment_name not in configs:
            msg = f"Config for '{experiment_name=}' not found!"
            maybe = difflib.get_close_matches(experiment_name, configs.keys())
            if maybe:
                msg += f" Did you mean '{maybe[0]}'?"
            raise ValueError(msg)

        self.config = configs[experiment_name]
        self.base_directory = base_directory
        self.dataset = dataset
        self.base_network = base_network
        self.experiment_name = experiment_name
        self.save_dir = Path(base_directory, dataset, base_network, experiment_name)

    def get_datamodule(self, **data_config_overrides):
        """The PytorchLightning datamodule for the experiment.
        Need to set up before you can use. E.g.:

        Example
        -------
        >>> exp = Experiment(...)
        >>> datamodule = exp.get_datamodule()
        >>> datamodule.setup("val")
        >>> val_dataloader = datamodule.val_dataloader()
        """
        # not a hard dependency
        from bcos.data.datamodules import ClassificationDataModule

        data_config = self.config["data"]
        data_config.update(data_config_overrides)
        registry = ClassificationDataModule.registry()
        if self.dataset in registry:
            datamodule = registry[self.dataset](data_config)
        else:
            available = list(registry.keys())
            _line_break = "\n" + " " * len("ValueError: ")
            raise ValueError(
                f"Dataset '{self.dataset}' not found! Available: {available}{_line_break}"
                f"If you're adding a new dataset, subclass ClassificationDataModule{_line_break}"
                f"in bcos.data.datamodules and make sure it ends with 'DataModule'{_line_break}"
                "(e.g. 'MyDatasetDataModule') (the suffix will be stripped)."
            )

        return datamodule

    @property
    def last_training_checkpoint_path(self) -> Path:
        """
        Get the path of the last checkpoint for this experiment.
        """
        return get_last_checkpoint_path_in_save_dir(self.save_dir)

    @property
    def last_training_checkpoint_state_dict(self) -> Dict[str, Any]:
        """
        Try to load the last checkpoint state dict.
        """
        save_path = self.last_training_checkpoint_path
        return device_safe_load_state_dict_from_path(save_path)

    def get_model(self, **kwargs: Any):
        """
        Create a new instance of the model for the experiment.

        Parameters
        ----------
        **kwargs
            Any additional keyword arguments to pass to the model constructor.

        Returns
        -------
        torch.nn.Module
            The model.
        """
        model_config = self.config["model"]
        model_config = update_config(model_config, kwargs)
        return self._get_model(model_config)

    def load_metrics(self) -> Metrics:
        """
        Returns
        -------
        Metrics
            Load a dictionary containing metrics from the last state dict.
            Tensor shape: [num_epochs (+1), 2] with columns (#epoch, metric_value)
            You can use the `.best_epoch_and_accuracy` attribute of the returned
            dict to get tuple containing the best epoch and accuracy. (See also
            `.get_best_epoch_and_metric_value_for(key)` method.)
        """
        return Metrics.from_experiment_dir(self.save_dir)

    def load_trained_model(
        self,
        reload: str = "last",
        ema: bool = False,
        verbose: bool = False,
        return_training_ckpt_if_possible: bool = False,
        **kwargs: Any,
    ) -> Union[torch.nn.Module, Dict[str, Union[torch.nn.Module, Dict[str, Any]]]]:
        """
        Load a trained model.

        Parameters
        ----------
        reload : str
            What weights to reload. Either "last", "best", "best_any" or "epoch_<N>".
            "best_any": will check both non-ema or ema (if they exist) weights for best accuracy
            "best" will load best ema weights if `ema=True` and best non-ema otherwise
            Default: "last"
        ema : bool
            Only consider EMA weights. Default: False
            Not applicable if `reload="best_any"`.
        verbose : bool
            Print information about loaded epoch and top-1 accuracy.
            Default: False
        return_training_ckpt_if_possible: bool
            Whether to ALSO return the whole training checkpoint dict instead.
            If `True`, then the return type will always be a dictionary with keys
            "model" and "ckpt" (for model and checkpoint respectively).
            Default: False
        **kwargs: Any
            Any additional keyword arguments to override the model config.

        Returns
        -------
        torch.nn.Module | Dict[str, torch.nn.Module | Dict[str, Any]]
            A trained model.
            Or a dictionary containing the model and the training checkpoint dict
            (with keys "model" and "ckpt" respectively).
        """
        model = self.get_model(**kwargs)

        # check if we have training checkpoints
        if (
            not self.save_dir.exists()
            or len(self.available_checkpoints(suppress_warnings=True)) == 0
        ):
            if verbose:
                print("No checkpoints found! Trying to load external weights...")
            if return_training_ckpt_if_possible:
                warnings.warn("No checkpoints found! Returning model only.")
            try:
                # if no training checkpoints available, then we try to load external weights
                model = self._try_load_external_weights(verbose, reload=reload, ema=ema)
                return (
                    model
                    if not return_training_ckpt_if_possible
                    else dict(
                        model=model,
                        ckpt=None,
                    )
                )
            except ModelFactoryDoesNotSupportPretrainedError:
                raise FileNotFoundError(
                    "No checkpoints found (and no external weights either)!"
                )

        # have training checkpoints, so load them
        state_dict, training_ckpt = get_state_dict_and_training_ckpt_from_save_dir(
            self.save_dir,
            reload=reload,
            ema=ema,
            verbose=verbose,
        )
        model.load_state_dict(state_dict)

        if return_training_ckpt_if_possible:
            return dict(model=model, ckpt=training_ckpt)
        else:
            return model

    def available_checkpoints(self, suppress_warnings: bool = False) -> List[Path]:
        """
        Returns a list to paths to available checkpoint files if any.

        Parameters
        ----------
        suppress_warnings : bool
            Whether to suppress warnings if no checkpoints are found.
            Default: False

        Returns
        -------
        List[Path]
            A list of paths to available checkpoint files.
        """
        if not self.save_dir.exists():
            raise ValueError(f"Save directory '{self.save_dir}' does not exist!")

        result = list(p for p in self.save_dir.iterdir() if p.suffix == ".ckpt")

        if len(result) == 0 and not suppress_warnings:
            warnings.warn(f"No checkpoints found in '{self.save_dir}'!")
        return result

    def _try_load_external_weights(
        self, verbose: bool, **kwargs: Any
    ) -> torch.nn.Module:
        """
        Try to load external weights via model factory.

        Parameters
        ----------
        verbose : bool
            Whether to print information about loaded epoch and top-1 accuracy.
        kwargs : Any
            Keyword arguments to check for `ema` and `reload` (which will be ignored)
            to inform the user about it.

        Returns
        -------
        torch.nn.Module
            Model loaded with (external) weights from the model factory.
        """
        if kwargs.get("ema", False):
            warnings.warn("ema=True will be ignored for external weights!")
        if kwargs.get("reload", "last") != "last":
            warnings.warn(
                "Seems like you're using a non-default value of reload. "
                "This will be ignored for external weights!"
            )

        # use inspect to check if the model factory has a `pretrained` argument
        # if it does, then we pass it to the model factory
        # otherwise, we just pass it to the model constructor
        model_factory = self._get_model
        model_factory_signature = inspect.signature(model_factory)
        if "pretrained" in model_factory_signature.parameters:
            model_config = self.config["model"]
            model = model_factory(model_config, pretrained=True)
        else:
            if verbose:
                print("No pretrained argument found in model factory!")
            raise ModelFactoryDoesNotSupportPretrainedError

        if verbose:
            print("Loaded external weights!")
        return model
