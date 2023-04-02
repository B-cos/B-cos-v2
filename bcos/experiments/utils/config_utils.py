"""
Utilities for working with configs.
"""
import argparse
import copy
import difflib
import warnings
from importlib import import_module
from typing import Any, Callable, Dict, List, Tuple, Union

import torch.nn

try:
    from rich import print as pprint
except ImportError:
    from pprint import pprint

from .structure_constants import (
    BASE_EXPERIMENTS_DIRECTORY,
    CONFIGS_MODULE,
    CONFIGS_VAR_NAME,
    MODEL_FACTORY_MODULE,
    MODEL_FACTORY_VAR_NAME,
    ROOT,
)

__all__ = [
    "update_config",
    "configs_cli",
    "get_configs_and_model_factory",
    "sanitize_config",
    "ALLOWED_SANITIZED_DATA_TYPES",
    # specific config creators
    "create_configs_with_different_seeds",
]


def update_config(old_config: Dict, new_config: Dict) -> Dict:
    """
    Creates a new updated config. The old config is copied and updated
    with the newer one. This is done recursively.

    Parameters
    ----------
    old_config : Dict
        The old config.
    new_config : Dict
        The new config to create an updated version of the old one.

    Returns
    -------
    Dict
        The updated version of the old config different from the old one.
    """
    result = copy.deepcopy(old_config)
    for k, v in new_config.items():
        # if subconfig update that recursively
        if k in result and isinstance(result[k], dict):
            assert isinstance(
                v, dict
            ), "Trying to overwrite a dict with something in a config!"
            result[k] = update_config(old_config=result[k], new_config=v)
        else:
            result[k] = v

    return result


def configs_cli(configs: Dict, *argv: str) -> None:
    """
    A CLI for working with configs. Primarily used for debugging.

    Parameters
    ----------
    configs : Dict[str, Dict[str, Any]]
        The configs to work with.
    argv: str
        Optionally, provided arguments to parse directly
        (do not use sys.argv).
    """
    parser = argparse.ArgumentParser(
        "Print config information. By default prints number of configs."
    )
    parser.add_argument(
        "-f",
        "--find",
        type=str,
        default=None,
        help="Check if given config is present and print it.",
    )
    parser.add_argument(
        "-s",
        "--to_script",
        action="store_true",
        default=False,
        help="Create a script file with commands for all experiments.",
    )
    parser.add_argument(
        "-a",
        "--print-all",
        action="store_true",
        default=False,
        help="Print all the names of the configs present.",
    )
    argv = None if len(argv) == 0 else argv
    args = parser.parse_args(argv)

    if len(configs) == 0:
        warnings.warn("No configs found. It's empty!")

    if args.to_script:
        cmd_template = (
            "python run_with_submitit.py --gpus $NUMGPUS --nodes $NUMNODES --timeout 4 --dataset ImageNet "
            "--base_network $BASENET --distributed --csv_logger --wandb_logger "
            "--wandb_project $WANDPROJ --experiment_name {exp_name} --amp --cache_dataset shm"
        )
        with open("run_exps.sh", "w") as file:
            for config in configs:
                file.write(cmd_template.format(exp_name=config) + "\n")
        return

    if args.find is not None:
        if args.find in configs:
            print(f"Found '{args.find}'")
            pprint(configs[args.find])
        else:
            print(f"No config named '{args.find}'!")
            maybe_alternative = difflib.get_close_matches(
                args.find, configs.keys(), n=1
            )
            if maybe_alternative:
                print(f"Did you mean '{maybe_alternative[0]}'?")
    elif args.print_all:
        for name in configs.keys():
            print(name)
    else:
        print(f"There are a total of {len(configs)} configs.")


def get_configs_and_model_factory(
    dataset: str, base_network: str
) -> Tuple[Dict, Callable[..., torch.nn.Module]]:
    """
    Gets all the configs and the model factory function for given
    base network.

    Parameters
    ----------
    dataset : str
        The dataset for the model.
    base_network : str
        The base network.

    Returns
    -------
    A tuple of (configs, model_factory).
    """
    base_module_path = ".".join(
        [ROOT, BASE_EXPERIMENTS_DIRECTORY, dataset, base_network]
    )
    model_path = ".".join([base_module_path, MODEL_FACTORY_MODULE])
    configs_path = ".".join([base_module_path, CONFIGS_MODULE])

    try:
        model_module = import_module(model_path)
    except ModuleNotFoundError:
        print(f"Unable to import '{model_path}'")
        raise
    try:
        configs_module = import_module(configs_path)
    except ModuleNotFoundError:
        print(f"Unable to import '{configs_path}'")
        raise

    return getattr(configs_module, CONFIGS_VAR_NAME), getattr(
        model_module, MODEL_FACTORY_VAR_NAME
    )


ALLOWED_SANITIZED_DATA_TYPES = (str, int, float, bool, tuple, list, type(None))
SANITIZED_DICT_TYPE = Dict[
    str, Union[str, int, float, bool, tuple, list, type(None), "SANITIZED_DICT_TYPE"]
]


def sanitize_config(config_dict: Dict) -> SANITIZED_DICT_TYPE:
    """
    This sanitizes the config dict so that the hyperparameters can be nicely
    logged as strings or numbers or other primitive types except dict.
    In particular, with W&B. This works recursively.

    Parameters
    ----------
    config_dict : Dict[str, Any]
        The config to sanitize.

    Notes
    -----
    If a *value* object has the method `__to_config__`, then that will be
    called instead to get a **dictionary**, instead of just using `repr` on it.
    For an example, see the class `OptimizerFactory`.

    Returns
    -------
    SANITIZED_DICT_TYPE
        The sanitized config.
    """
    result = dict()
    for key, value in config_dict.items():
        if isinstance(value, dict):  # subconfig
            value = sanitize_config(value)
        elif isinstance(value, ALLOWED_SANITIZED_DATA_TYPES):  # one of allowed types
            pass  # do nothing already suitable
        elif hasattr(value, "__to_config__"):  # non-dict object's subconfig
            sub_config_dict = value.__to_config__()
            value = sanitize_config(sub_config_dict)
        else:  # just use repr if nothing else possible
            value = repr(value)

        result[key] = value

    return result


# --------------------------------------------------------------------------------
# config creators
# --------------------------------------------------------------------------------
def create_configs_with_different_seeds(
    configs: Dict[str, Dict[str, Any]], seeds: Union[List[int], int]
):
    """
    For each experiment config in given configs, changes the seed to given seed(s).

    Parameters
    ----------
    configs : Dict[str, Dict[str, Any]]
        The name to config dict.
    seeds: List[int] | int
        Alternatively provide multiple seeds.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Changed configs.
    """
    if isinstance(seeds, int):
        seeds = [seeds]

    result = dict()
    for seed in seeds:
        new_configs = copy.deepcopy(configs)
        new_configs = {f"{key}-{seed=}": value for key, value in new_configs.items()}
        for name, config in new_configs.items():
            config["seed"] = seed
        result.update(new_configs)

    return result
