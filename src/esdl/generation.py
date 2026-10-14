"""Seeded raw-domain generation, separate from serialization and simulation."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a nonnegative integer")


def generate_cylinders(count: int, seed: int = 0) -> NDArray[np.float64]:
    """Generate pairs of the notebook's 69-cell raster disks at permittivity 2.

    Each center's row and column are uniformly sampled from 4 through 23;
    overlapping disks merge. Background permittivity is one. No files are
    created, and the global NumPy random state is unaffected.
    """
    _validate_count(count)
    rng = np.random.default_rng(seed)
    domains = np.ones((count, 28, 28), dtype=np.float64)
    for domain in domains:
        rows = rng.integers(4, 24, size=2)
        columns = rng.integers(4, 24, size=2)
        for row, column in zip(rows, columns):
            domain[row - 4, column - 2 : column + 3] = 2
            domain[row - 3, column - 3 : column + 4] = 2
            domain[row - 2 : row + 3, column - 4 : column + 5] = 2
            domain[row + 3, column - 3 : column + 4] = 2
            domain[row + 4, column - 2 : column + 3] = 2
    return domains


def generate_mnist(
    count: int,
    split: Literal["train", "test"] = "train",
    seed: int = 0,
    download_dir: str | Path = "data/raw/mnist",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Sample MNIST with replacement and return raw domains, labels, indices.

    Torchvision is imported only when MNIST is requested. Its dataset loader
    downloads source MNIST files into ``download_dir`` when necessary; project
    domain/field arrays are never saved or overwritten here. Scaling uses the
    original float32 ``ToTensor`` precision before conversion to float64.
    """
    _validate_count(count)
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if count == 0:
        return np.empty((0, 28, 28), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError("MNIST generation requires torchvision") from exc

    dataset = MNIST(root=str(Path(download_dir).absolute()), train=split == "train", download=True)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=count, dtype=np.int64)
    pixels = dataset.data[indices].numpy().astype(np.float32)
    domains = (pixels / 255 + 1).astype(np.float64)
    labels = dataset.targets[indices].numpy().astype(np.float64)
    return domains, labels, indices
