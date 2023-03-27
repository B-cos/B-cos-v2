import argparse
import copy
import os
from pathlib import Path

import submitit

import bcos.training.trainer as trainer
import train


def parse_args():
    train_parser = train.get_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[train_parser])
    parser.add_argument("--gpus", default=4, type=int, help="Number of GPUs per node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes")
    parser.add_argument(
        "--timeout", default=None, type=float, help="Job duration in hours"
    )
    parser.add_argument(
        "--timeout_min", default=None, type=int, help="Job duration in minutes"
    )
    parser.add_argument("--job_name", type=str, help="Job name")
    parser.add_argument(
        "--partition",
        default="gpu16,gpu20,gpu22",
        type=str,
        help="Partition where to submit",
    )
    return parser.parse_args()


class RunExperiment:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        trainer.run_training(copy.deepcopy(self.args))

    def checkpoint(self):
        self.args.resume = True  # overwrite
        rejob = type(self)(copy.deepcopy(self.args))
        return submitit.helpers.DelayedSubmission(rejob)


def get_job_dir(args):
    save_dir = str(
        Path(
            args.base_directory,
            args.dataset,
            args.base_network,
            args.experiment_name,
            "slurm_logs",
        )
    ).rstrip("/")
    save_dir += "/%j"

    return save_dir


def submit_experiment():
    args = parse_args()
    train._args_validation(args)  # noqa

    if args.gpus > 1 or args.nodes > 1:
        args.distributed = True  # force ddp

    executor = submitit.AutoExecutor(
        folder=get_job_dir(args), slurm_max_num_timeout=300
    )

    if args.timeout_min is not None:
        assert args.timeout is None
        timeout = args.timeout_min + 4
    else:
        assert args.timeout_min is None
        timeout = int(args.timeout * 60) + 4

    executor.update_parameters(
        name=args.job_name or args.experiment_name,
        mem_gb=110 * args.gpus,
        gpus_per_node=args.gpus,
        tasks_per_node=args.gpus,
        cpus_per_task=16,
        nodes=args.nodes,
        timeout_min=timeout,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=4
        * 60,  # a lower time sometimes causes problems resuming on our cluster
    )

    if args.wandb_logger and args.wandb_id is None:
        import wandb

        wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
        print(f"Wandb run id: '{wandb_id}'")
        args.wandb_id = wandb_id

    experiment = RunExperiment(args)
    job = executor.submit(experiment)

    print("Submitted job id:", job.job_id)


if __name__ == "__main__":
    submit_experiment()
