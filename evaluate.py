import argparse
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731

from bcos.data.datamodules import ClassificationDataModule
from bcos.experiments.utils import Experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model.", add_help=add_help
    )

    # specify save dir and experiment config
    parser.add_argument(
        "--hubconf",
        help="Test model from local hubconf file.",
    )

    parser.add_argument(
        "--base_directory",
        default="./experiments",
        help="The base directory.",
    )
    parser.add_argument(
        "--dataset",
        choices=["ImageNet", "CIFAR10"],
        default="ImageNet",
        help="The dataset.",
    )
    parser.add_argument(
        "--base_network", help="The model config or base network to use."
    )
    parser.add_argument("--experiment_name", help="The name of the experiment to run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reload", help="What ckpt to load. ['last', 'best', 'epoch_<N>', 'best_any']"
    )
    group.add_argument(
        "--weights",
        metavar="PATH",
        type=Path,
        help="Specific weight state dict to load.",
    )

    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load the EMA stored version if it exists. Not applicable for reload='best_any'.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to use. Default is 1"
    )
    parser.add_argument(
        "--no-cuda",
        default=False,
        action="store_true",
        help="Force into not using cuda.",
    )

    return parser


def run_evaluation(args):
    global device
    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # get model, config, and data
    model, config = load_model_and_config(args)
    test_loader = get_test_loader(args.dataset, config)

    # do evaluation
    evaluate(model, test_loader)


def evaluate(model, data_loader):
    # https://github.com/pytorch/vision/blob/657c0767c5ca5564c8b437ac44263994c8e0/references/classification/train.py#L61
    model.eval()

    total_samples = 0
    total_correct_top1 = 0
    total_correct_top5 = 0
    with torch.inference_mode():
        for image, target in tqdm(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)

            total_samples += image.shape[0]
            correct_top1, correct_top5 = check_correct(output, target, topk=(1, 5))
            total_correct_top1 += correct_top1.item()
            total_correct_top5 += correct_top5.item()

    acc1 = total_correct_top1 / total_samples
    acc5 = total_correct_top5 / total_samples
    print(
        f"Out of a total of {total_samples}, got {total_correct_top1=} and {total_correct_top5=}"
    )
    print()
    print("--------------------------------------------")
    print(f"Acc@1 {acc1:.3%} Acc@5 {acc5:.3%}")
    print("--------------------------------------------")
    print()


def check_correct(output, target, topk=(1,)):
    with torch.inference_mode():
        maxk = max(topk)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum()
            res.append(correct_k)
        return res


def load_model_and_config(args):
    # a bit messy because of trying to directly use hubconf
    if args.hubconf is not None:
        import hubconf

        model = getattr(hubconf, args.hubconf)(pretrained=True)
        config = model.config
    else:
        experiment = Experiment(
            base_directory=args.base_directory,
            path_or_dataset=args.dataset,
            base_network=args.base_network,
            experiment_name=args.experiment_name,
        )
        config = experiment.config

        if args.reload is not None:
            model = experiment.load_trained_model(
                reload=args.reload, verbose=True, ema=args.ema
            )
        elif args.weights is not None:
            model: torch.nn.Module = experiment.get_model()
            state_dict = torch.load(args.weights, map_location="cpu")
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(
                    "Error loading state dict. Please note that --weights only supports "
                    "loading model state dict and not from training checkpoints."
                )
                raise e
        else:
            raise RuntimeError(
                "One of --reload, --weights or --hubconf must be provided!"
            )

    model = model.to(device)

    return model, config


def get_test_loader(dataset, config):
    registry = ClassificationDataModule.registry()
    if dataset in registry:
        datamodule = registry[dataset](config)
    else:
        available_datasets = list(registry.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset}'. Available datasets are: {available_datasets}"
        )

    # get data and set batchsize
    datamodule.batch_size = args.batch_size
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    return test_loader


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_evaluation(args)
