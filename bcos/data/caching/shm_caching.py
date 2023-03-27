"""
Caching into SHM.
"""
import os
from pathlib import Path

import bcos.settings as settings


def cache_tar_files_to_shm() -> None:
    """
    Caches imagenet tar files into shm.
    Intended to work only for imagenet.
    """
    IMAGENET_TRAIN_TAR_FILES_DIR = settings.IMAGENET_TRAIN_TAR_FILES_DIR
    SHMTMPDIR = settings.SHMTMPDIR
    SHMTMPDIR = Path(SHMTMPDIR)
    TRAIN_DIRECTORY = SHMTMPDIR / "train"

    tar_files_to_extract = list(Path(IMAGENET_TRAIN_TAR_FILES_DIR).iterdir())
    assert len(tar_files_to_extract) == 1_000

    TRAIN_DIRECTORY.mkdir(exist_ok=True)

    os.system(f"df -H {SHMTMPDIR}")
    os.system(
        f"echo {' '.join(map(str, tar_files_to_extract))} "
        f"| tr ' ' '\n' "
        f"| xargs -P {os.cpu_count()} "
        "-I {} cp {} "
        f"{SHMTMPDIR}"
    )
    os.system(
        f"echo {' '.join([str(SHMTMPDIR / p.name) for p in tar_files_to_extract])} "
        f"| tr ' ' '\n' "
        f"| xargs -P {os.cpu_count()} "
        "-I {} tar xf {}"
        f" --directory {TRAIN_DIRECTORY}"
    )
    os.system(
        f"echo {' '.join([str(SHMTMPDIR / p.name) for p in tar_files_to_extract])} "
        f"| tr ' ' '\n' "
        f"| xargs -P {os.cpu_count()} -n 8 rm -rf"
    )
    os.system(f"df -H {SHMTMPDIR}")
