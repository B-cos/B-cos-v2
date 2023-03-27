import argparse
import difflib
import os

from bcos.experiments.utils import get_configs_and_model_factory
from bcos.training import trainer


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Start training.", add_help=add_help)

    # specify save dir and experiment config
    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory to store to.",
    )
    parser.add_argument(
        "--dataset", choices=["ImageNet", "CIFAR10"], help="The dataset."
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    # other training args
    parser.add_argument(
        "--track_grad_norm",
        default=False,
        action="store_true",
        help="Track the L_2 norm of the gradient.",
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="Use distributed mode.",
    )
    parser.add_argument(
        "--force-no-resume",
        dest="resume",
        default=True,  # so by default always resume (notice dest!)
        action="store_false",  # if given do not resume!
        help="Force restart/retrain experiment.",
    )
    parser.add_argument(
        "--amp", default=False, action="store_true", help="Use mixed precision."
    )
    parser.add_argument(
        "--jit",
        default=False,
        action="store_true",
        help="Use torch.jit.script on the model.",
    )
    parser.add_argument(
        "--cache_dataset",
        default=None,
        choices=["onthefly", "shm"],
        help="Cache dataset.",
    )
    parser.add_argument(
        "--refresh_rate",
        type=int,
        help="Refresh rate for progress bar.",
    )

    # loggers
    parser.add_argument(
        "--csv_logger", action="store_true", default=False, help="Use CSV logger."
    )
    parser.add_argument(
        "--tensorboard_logger",
        action="store_true",
        default=False,
        help="Use tensorboard logger.",
    )

    parser.add_argument(
        "--wandb_logger", action="store_true", default=False, help="Use WB logger."
    )
    parser.add_argument(
        "--wandb_project",
        # here so that custom args validation doesn't complain
        default=os.getenv("WANDB_PROJECT"),
        help="Project name of run.",
    )
    parser.add_argument(
        "--wandb_id", default=os.getenv("WANDB_ID"), help="Project name of run."
    )
    parser.add_argument(
        "--wandb_name",
        default=None,  # use args.experiment_name
        help="Override wandb exp. name. Default use --experiment_name",
    )

    # explanations logging
    parser.add_argument(
        "--explanation_logging",
        action="store_true",
        dest="explanation_logging",
        default=False,
        help="Enable explanation logging.",
    )
    parser.add_argument(
        "--explanation_logging_every_n_epochs",
        type=int,
        default=1,
        help="Log explanations every n epochs.",
    )

    # debugging stuff
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use trainer's fast dev run mode.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging mode.",
    )

    return parser


def _args_validation(args):
    # check if config exists
    configs, _ = get_configs_and_model_factory(args.dataset, args.base_network)
    if args.experiment_name not in configs:
        err_msg = f"Unknown config '{args.experiment_name}'!"
        possible = difflib.get_close_matches(args.experiment_name, configs.keys())
        if possible:
            err_msg += f" Did you mean '{possible[0]}'?"
        raise RuntimeError(err_msg)

    # check for resume
    assert hasattr(args, "resume"), "no resume arg in args!"

    # stay organized
    if args.wandb_logger:
        assert args.wandb_project is not None, "Provide a project name for WB!"


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    _args_validation(args)
    try:
        trainer.run_training(args)
    except Exception:
        import pdb

        if args.debug:
            pdb.post_mortem()
        raise
