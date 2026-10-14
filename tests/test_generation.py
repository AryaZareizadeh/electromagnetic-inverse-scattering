"""Deterministic raster generation and real local MNIST decoding."""

import struct

import numpy as np
import pytest


def test_cylinders_are_repeatable_and_have_original_raster_geometry():
    from esdl.generation import generate_cylinders

    result = generate_cylinders(40, seed=42)
    assert result.shape == (40, 28, 28)
    np.testing.assert_array_equal(result, generate_cylinders(40, seed=42))
    assert not np.array_equal(result, generate_cylinders(40, seed=43))
    assert set(np.unique(result)) == {1, 2}
    counts = np.count_nonzero(result > 1, axis=(1, 2))
    assert np.all((counts >= 69) & (counts <= 138))
    # Two non-overlapping original disks have 138 cells, each disk 69.
    sample = result[np.flatnonzero(counts == 138)[0]]
    visited = np.zeros_like(sample, dtype=bool)
    component_sizes = []
    for start in np.argwhere(sample == 2):
        if visited[tuple(start)]:
            continue
        pending = [tuple(start)]
        count = 0
        while pending:
            row, column = pending.pop()
            if not (0 <= row < 28 and 0 <= column < 28):
                continue
            if visited[row, column] or sample[row, column] != 2:
                continue
            visited[row, column] = True
            count += 1
            pending.extend([(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)])
        component_sizes.append(count)
    assert component_sizes == [69, 69]


def test_zero_cylinders_has_empty_sample_axis():
    from esdl.generation import generate_cylinders

    assert generate_cylinders(0).shape == (0, 28, 28)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_cylinder_count_is_rejected(count):
    from esdl.generation import generate_cylinders

    with pytest.raises(ValueError):
        generate_cylinders(count)


@pytest.mark.parametrize("kwargs", [{"count": -1}, {"count": 1.5}, {"count": 1, "split": "unknown"}])
def test_invalid_mnist_request_is_rejected_before_loading_torchvision(kwargs):
    from esdl.generation import generate_mnist

    with pytest.raises(ValueError):
        generate_mnist(**kwargs)


def test_mnist_uses_real_local_idx_data_and_preserves_original_pixel_precision(tmp_path):
    pytest.importorskip("torchvision")
    from esdl.generation import generate_mnist

    # Real MNIST raw files exercise torchvision's decoder without a download.
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True)
    images = np.zeros((3, 28, 28), dtype=np.uint8)
    images[1, 4, 7] = 128
    images[2, 12, 19] = 255
    labels = np.array([2, 4, 8], dtype=np.uint8)
    for split in ("train", "t10k"):
        (raw / f"{split}-images-idx3-ubyte").write_bytes(
            struct.pack(">IIII", 2051, 3, 28, 28) + images.tobytes()
        )
        (raw / f"{split}-labels-idx1-ubyte").write_bytes(
            struct.pack(">II", 2049, 3) + labels.tobytes()
        )
    result, result_labels, indices = generate_mnist(20, split="test", seed=7, download_dir=tmp_path)
    assert result.shape == (20, 28, 28)
    assert result.dtype == np.float64
    assert indices.shape == (20,)
    assert indices.dtype.kind in "iu"
    assert set(indices) == {0, 1, 2}
    expected = (images[indices].astype(np.float32) / 255 + 1).astype(np.float64)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result_labels, labels[indices])
    repeated = generate_mnist(20, split="test", seed=7, download_dir=tmp_path)
    for first, second in zip((result, result_labels, indices), repeated):
        np.testing.assert_array_equal(first, second)
