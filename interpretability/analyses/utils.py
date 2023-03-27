import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

from interpretability.explanation_methods.utils import ExplainerImportFailedWarning


class Analyser:
    default_config = {}

    def __init__(self, experiment, **config):
        self.experiment = experiment
        for k, v in self.default_config.items():
            if k not in config:
                config[k] = v
        self.config = config
        self.results = None

    def analysis(self):
        raise NotImplementedError("Need to implement analysis function.")

    def run(self):
        START = time.perf_counter()

        results = self.analysis()
        self.save_results(results)

        print(f"Took time: {time.perf_counter() - START:,.2f}s")

    def save_results(self, results):
        save_dir: Path = self.experiment.save_dir / self.get_save_folder()
        save_dir.mkdir(parents=True, exist_ok=True)

        for k, v in results.items():
            np.savetxt(save_dir / "{}.np".format(k), v)

        with (save_dir / "config.log").open("w") as file:
            for k, v in self.get_config().items():
                k_v_str = "{k}: {v}".format(k=k, v=v)
                print(k_v_str)
                file.writelines([k_v_str, "\n"])

            # add git commit if in a git repo
            if shutil.which("git") is not None:
                try:
                    git_commit = (
                        subprocess.check_output(["git", "rev-parse", "HEAD"])
                        .decode("utf-8")
                        .strip()
                    )
                    print(f"git_commit: {git_commit}")
                    file.writelines([f"git_commit: {git_commit}", "\n"])
                except subprocess.CalledProcessError:
                    warnings.warn("Could not get git commit to save in config.log!")

        print(f"Saved results to '{save_dir}'")

    def get_save_folder(self, epoch=None):
        raise NotImplementedError("Need to implement get_save_folder function.")

    def get_loaded_epoch(self):
        raise NotImplementedError("Need to implement the get_loaded_epoch function.")

    def get_config(self):
        config = self.config
        config.update({"epoch": self.get_loaded_epoch()})
        return config

    def load_results(self, epoch=None):
        save_path: Path = self.experiment.save_dir / self.get_save_folder(epoch)
        # print("Trying to load results from", save_path)
        if not save_path.exists():
            return
        results = dict()
        for file in save_path.iterdir():
            if file.suffix != ".np":
                continue
            results[file.stem] = np.loadtxt(file)
        self.results = results


def get_explainer_factory(supress_import_warnings=True):
    with warnings.catch_warnings():
        if supress_import_warnings:
            warnings.simplefilter("ignore", ExplainerImportFailedWarning)
        from interpretability.explanation_methods.explainers import get_explainer

        return get_explainer
