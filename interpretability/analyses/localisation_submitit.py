import argparse
import functools
import itertools

import submitit

from bcos.experiments.utils import Experiment

from .localisation import LocalisationAnalyser, argument_parser


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[argument_parser(multiple_args=True, add_help=False)]
    )
    parser.add_argument(
        "--timeout_min", required=True, type=int, help="Job duration in minutes"
    )
    parser.add_argument("--job_name", type=str, help="Job name")
    parser.add_argument(
        "--log_folder",
        default="slurm_logs/localisation/%j",
        type=str,
        help="Folder to store slurm logs, submission file etc.",
    )
    parser.add_argument(
        "--partition",
        default="gpu16,gpu20,gpu22",
        type=str,
        help="Partition where to submit",
    )
    return parser.parse_args()


def run_localisation_job(config, other):
    save_path, smooth, analysis_config = other
    experiment = Experiment(save_path)

    analyser = LocalisationAnalyser(
        experiment,
        analysis_config,
        force_confidence_calculation=config.force_confidence_calculation,
        explainer_name=config.explainer_name,
        explainer_config=config.explainer_config,
        smooth=smooth,
        reload=config.reload,
        ema=config.ema,
        batch_size=config.batch_size,
    )
    analyser.run()


def submit_localisation_jobs():
    args = parse_args()

    executor = submitit.AutoExecutor(folder=args.log_folder)
    executor.update_parameters(
        mem_gb=32 * 1,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=16,
        nodes=1,
        timeout_min=args.timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
    )
    if args.job_name is not None:
        executor.update_parameters(
            name=args.job_name,
        )

    # submit array job
    jobs = executor.map_array(
        functools.partial(run_localisation_job, args),
        itertools.product(args.save_path, args.smooth, args.analysis_config),
    )

    print("Number of submitted jobs:", len(jobs))


if __name__ == "__main__":
    submit_localisation_jobs()
