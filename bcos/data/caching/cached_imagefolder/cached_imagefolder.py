"""
Convert a dataset into a in-memory cached dataset.
We use https://tutorials.baguasys.com/more-optimizations/cached-dataset

We only store loaded images to prevent high IO.
From https://github.com/BaguaSys/bagua
"""
import torch.utils.data as data
from torchvision.datasets import ImageFolder

from .cached_loader import CacheLoader

DEFAULT_CAPACITY = 200 * (1024**3)


class CachedImageFolder(data.Dataset):
    def __init__(self, underlying: ImageFolder, capacity: int = DEFAULT_CAPACITY):
        super().__init__()

        self.underlying = underlying
        self.transform = self.underlying.transform
        self.target_transform = self.underlying.target_transform
        self.samples = self.underlying.samples
        self.loader = self.underlying.loader

        self.cache_loader = CacheLoader(
            backend="redis",
            capacity_per_node=capacity,
            hosts=None,
            cluster_mode=True,
        )

    def __len__(self):
        return len(self.underlying)

    def __getitem__(self, index):
        # based on ImageFolder.__getitem__
        path, target = self.samples[index]
        sample = self.cache_loader.get(path, self.loader)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
