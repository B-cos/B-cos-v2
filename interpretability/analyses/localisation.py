import argparse
import pickle
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bcos.experiments.utils import Experiment
from interpretability.analyses.localisation_configs import configs
from interpretability.analyses.utils import Analyser, get_explainer_factory

get_explainer = get_explainer_factory(supress_import_warnings=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalisationAnalyser(Analyser):
    default_config = {"explainer_name": "Ours", "explainer_config": None}
    conf_fn = "conf_results.pkl"

    def __init__(
        self,
        experiment: Experiment,
        config_name: str,
        plotting_only: bool = False,
        verbose: bool = True,
        force_confidence_calculation: bool = False,
        **config,
    ):
        """
        This analyser evaluates the localisation metric (see CoDA-Net paper).
        Args:
            experiment: Experiment object.
            config_name: name of the localisation config to use
            plotting_only: Whether or not to load previous results. These can then be used for plotting.
            **config:
                explainer_config: Config key for the explanation configurations.
                explainer_name: Which explanation method to load. Default is Ours.
                verbose: Warn when overwriting passed parameters with the analysis config parameters.

        """
        self.config_name = config_name
        analysis_config = configs[config_name]
        if verbose:
            for k in analysis_config:
                if k in config:
                    print(
                        "CAVE: Overwriting parameter:",
                        k,
                        analysis_config[k],
                        config[k],
                        flush=True,
                    )
        analysis_config.update(config)
        super().__init__(experiment=experiment, **analysis_config)
        if plotting_only:
            self.load_results()
            return

        model_data = experiment.load_trained_model(
            reload=self.config["reload"],
            verbose=verbose,
            ema=self.config["ema"],
            return_training_ckpt_if_possible=True,
        )
        model = model_data["model"]

        # set in evaluation mode...
        model = model.eval()
        self.model = model.to(device, non_blocking=True)

        # get epoch
        if "ckpt" in model_data and model_data["ckpt"] is not None:
            self._epoch = model_data["ckpt"]["epoch"] + 1
        else:
            self._epoch = None

        self.explainer = get_explainer(
            model,
            self.config["explainer_name"],
            self.config["explainer_config"],
            batch_size=self.config["batch_size"],
        )
        self.sorted_confs: "Optional[dict[int, list[tuple[int, float]]]]" = None
        """A mapping from class id to list of (image_idx, confidence) tuples"""

        self._base_save_folder = experiment.save_dir
        self.compute_sorted_confs(
            force_confidence_calculation=force_confidence_calculation
        )

    def get_loaded_epoch(self) -> Optional[int]:
        return self._epoch

    @property
    def save_folder(self) -> Path:
        save_folder = self._base_save_folder / self.get_save_folder(
            self.get_loaded_epoch()
        )
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=True)
        return save_folder

    @property
    def sorted_confs_file_path(self) -> Path:
        save_folder = (
            self._base_save_folder
            / "localisation_analysis"
            / f"epoch_{self.get_loaded_epoch()}"
        )
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=True)
        return save_folder / self.conf_fn

    def compute_sorted_confs(self, force_confidence_calculation: bool = False) -> None:
        """
        Sort image indices by the confidence of the classifier and store in sorted_confs.
        Returns: None

        """
        fp = self.sorted_confs_file_path

        if fp.exists():
            if not force_confidence_calculation:
                print("Loading stored confidences", flush=True)
                with fp.open("rb") as file:
                    self.sorted_confs = pickle.load(file)
                return  # no need to re-calculate
            else:
                print("Forcefully re-calculating confidences now!", flush=True)
        else:
            print("No confidences file found, calculating now.", flush=True)

        datamodule = (
            self.experiment.get_datamodule()
        )  # no need to resize for confidences
        datamodule.setup("test")
        confidences = {i: [] for i in range(datamodule.NUM_CLASSES)}

        datamodule.batch_size = self.config["batch_size"]
        loader = datamodule.test_dataloader()
        img_idx = -1
        with torch.inference_mode():
            for img, tgt in tqdm(loader, desc="Calculating confidences"):
                img = img.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                logits, classes = self.model(img).max(1)
                for logit, pd_class, gt_class in zip(logits, classes, tgt):
                    img_idx += 1
                    if pd_class != gt_class:
                        continue  # wrongly classified do not take
                    confidences[int(gt_class.item())].append((img_idx, logit.item()))

        for k, vlist in confidences.items():
            confidences[k] = sorted(vlist, key=lambda x: x[1], reverse=True)

        with fp.open("wb") as file:
            pickle.dump(confidences, file)

        self.sorted_confs = confidences

    def get_sorted_indices(self) -> list:
        """
        This method generates a list of indices to be used for sampling from the dataset and evaluating the
            multi images.
        In particular, the images per class are sorted by their confidence.
        Then, a random set of n classes (for the multi image) is sampled and for each class the next
            most confident image index that was not used yet is added to the list.
        Thus, when using this list for creating multi images, the list contains blocks of size n with
        image indices such that (1) each class occurs at most once per block and (2) the class confidences
            decrease per block for each class individually.

        Returns: list of indices

        """
        idcs = []
        classes = np.array([k for k in self.sorted_confs.keys()])
        class_indexer = {k: 0 for k in classes}

        # Only use images with a minimum confidence of 50%
        # This is, of course, the same for every attribution method
        def get_conf_mask_v(_c_idx: int) -> bool:
            return (
                torch.tensor(self.sorted_confs[_c_idx][class_indexer[_c_idx]][1])
                .sigmoid()
                .item()
                > self.config["conf_thresh"]
            )

        # Only use classes that are still confidently classified
        mask = np.array(
            [
                False if len(self.sorted_confs[k]) == 0 else get_conf_mask_v(k)
                for k in classes
            ]
        )
        n_imgs = self.config["n_imgs"]
        # Always use the same set of classes for a particular model
        np.random.seed(42)
        while mask.sum() > n_imgs:
            # Of the still available classes, sample a set of classes randomly
            sample = np.random.choice(classes[mask], size=n_imgs, replace=False)

            for c_idx in sample:
                # Store the corresponding index of the next class image for each of the randomly sampled classes
                img_idx, conf = self.sorted_confs[c_idx][class_indexer[c_idx]]
                class_indexer[c_idx] += 1
                mask[c_idx] = (
                    get_conf_mask_v(c_idx)
                    if class_indexer[c_idx] < len(self.sorted_confs[c_idx])
                    else False
                )
                idcs.append(img_idx)
        return idcs

    def get_save_folder(self, epoch: Optional[int] = None):
        """
        'Computes' the folder in which to store the results.
        Args:
            epoch: currently evaluated epoch.

        Returns: Path to save folder.

        """
        if epoch is None:
            epoch = self.get_loaded_epoch()
        return join(
            "localisation_analysis",
            "epoch_{}".format(epoch),
            self.config_name,
            self.config["explainer_name"],
            "smooth-{}".format(int(self.config["smooth"])),
            self.config["explainer_config"],
        )

    def analysis(self) -> Dict[str, Any]:
        sample_size = self.config["sample_size"]
        n_imgs = self.config["n_imgs"]
        assert (
            n_imgs**0.5
        ).is_integer(), f"{n_imgs=} must be a perfect square but isn't!"
        smooth = self.config["smooth"]

        data_config_overrides = {}
        if self.config["do_rescale"]:
            previous_transform = self.experiment.config["data"]["test_transform"]
            n = int(n_imgs**0.5)
            crop_size = previous_transform.args["crop_size"] // n
            resize_size = previous_transform.args["resize_size"] // n
            new_transform = previous_transform.with_args(
                resize_size=resize_size, crop_size=crop_size
            )
            data_config_overrides["test_transform"] = new_transform
        datamodule = self.experiment.get_datamodule(**data_config_overrides)
        datamodule.setup("test")
        fixed_indices = self.get_sorted_indices()
        metric = []
        explainer = self.explainer
        offset = 0
        dataset = datamodule.test_dataloader().dataset
        single_shape = dataset[0][0].shape[-1]
        for count in tqdm(range(sample_size), desc="Analysis..."):
            multi_img, tgts, offset = self.make_multi_image(
                n_imgs, dataset, offset=offset, fixed_indices=fixed_indices
            )

            # calculate the attributions for all classes that are participating
            attributions = explainer.attribute_selection(multi_img, tgts).sum(
                1, keepdim=True
            )

            # Smooth the attributions
            if smooth:
                attributions = F.avg_pool2d(
                    attributions, smooth, stride=1, padding=(smooth - 1) // 2
                )

            # Only compare positive attributions
            attributions = attributions.clamp(min=0)

            # Calculate the relative amount of attributions per region. Use avg_pool for simplicity.
            with torch.no_grad():
                contribs = (
                    F.avg_pool2d(attributions, single_shape, stride=single_shape)
                    .permute(0, 1, 3, 2)
                    .reshape(attributions.shape[0], -1)
                )
                total = contribs.sum(1, keepdim=True)
            contribs = (
                torch.where(
                    total * contribs > 0, contribs / total, torch.zeros_like(contribs)
                )
                .detach()
                .cpu()
                .numpy()
            )
            metric.append([contrib[idx] for idx, contrib in enumerate(contribs)])
        result = np.array(metric).flatten()
        print()
        print(
            "Percentiles of localisation accuracy (25, 50, 75, 100): ",
            np.percentile(result, [25, 50, 75, 100]),
        )
        print()
        return {"localisation_metric": result}

    @staticmethod
    def make_multi_image(n_imgs, dataset, offset=0, fixed_indices=None):
        """
        From the offset position takes the next n_imgs that are of different classes according to the order in the
        dataset or fixed_indices .
        Args:
            n_imgs: how many images should be combined for a multi images
            dataset: the dataset
            offset: current offset
            fixed_indices: whether or not to use pre-defined indices (e.g., first ordering images by confidence).

        Returns: the multi_image, the targets in the multi_image and the new offset


        """
        assert n_imgs in [4, 9]
        tgts = []
        imgs = []
        count = 0
        i = 0
        if fixed_indices is not None:
            mapper = fixed_indices
        else:
            mapper = list(range(len(dataset)))

        # Going through the dataset to sample images
        while count < n_imgs:
            img, tgt = dataset[mapper[i + offset]]
            i += 1
            tgt_idx = tgt
            # if the class of the new image is already added to the list of images for the multi-image, skip this image
            # This should actually not happen since the indices are sorted in blocks of 9 unique labels
            if tgt_idx in tgts:
                continue
            imgs.append(img[None])
            tgts.append(tgt_idx)
            count += 1
        img = torch.cat(imgs, dim=0)
        img = (
            img.view(-1, int(np.sqrt(n_imgs)), int(np.sqrt(n_imgs)), *img.shape[-3:])
            .permute(0, 3, 2, 4, 1, 5)
            .reshape(
                -1,
                img.shape[1],
                img.shape[2] * int(np.sqrt(n_imgs)),
                img.shape[3] * int(np.sqrt(n_imgs)),
            )
        )

        return img.to(device, non_blocking=True), tgts, i + offset + 1


def argument_parser(multiple_args=False, add_help=True):
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    nargs = "+" if multiple_args else None

    parser = argparse.ArgumentParser(
        description="Localisation metric analyser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "--save_path", default=None, nargs=nargs, help="Path for model checkpoints."
    )
    parser.add_argument(
        "--reload",
        default="last",
        type=str,
        help="Which epoch to load. Options are 'last', 'best', 'best_any' and 'epoch_X',"
        "as long as epoch_X exists.",
    )
    parser.add_argument(
        "--ema",
        default=False,
        action="store_true",
        help="Load EMA weights instead if they exist.",
    )
    parser.add_argument(
        "--explainer_name",
        default="Ours",
        type=str,
        help="Which explainer method to use. Ours uses trainer.attribute.",
    )
    parser.add_argument(
        "--analysis_config",
        default="500_3x3",
        nargs=nargs,
        type=str,
        help="Which analysis configuration file to load.",
    )
    parser.add_argument(
        "--explainer_config",
        default="default",
        type=str,
        help="Which explainer configuration file to load.",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for the data loader."
    )
    parser.add_argument(
        "--smooth",
        required=True,
        type=int,
        nargs=nargs,
        help="Determines by how much the attribution maps are smoothed (avg_pool).",
    )
    parser.add_argument(
        "--force_confidence_calculation",
        default=False,
        action="store_true",
        help="Forcefully recalculate the confidences, even if stored.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    return opts


def main(config):
    print("Starting localisation analysis for", config)
    print()
    print()

    experiment = Experiment(config.save_path)

    analyser = LocalisationAnalyser(
        experiment,
        config.analysis_config,
        force_confidence_calculation=config.force_confidence_calculation,
        explainer_name=config.explainer_name,
        explainer_config=config.explainer_config,
        smooth=config.smooth,
        reload=config.reload,
        ema=config.ema,
        batch_size=config.batch_size,
    )
    analyser.run()


if __name__ == "__main__":
    params = get_arguments()

    try:
        main(params)
    except Exception:
        if params.debug:
            import pdb

            pdb.post_mortem()
        raise
