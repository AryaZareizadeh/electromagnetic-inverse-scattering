"""Honest regression metrics for channel-first fields or flattened batches."""

import math

import numpy as np
import torch


def _metric_tensor(values, *, device=None):
    # torch.as_tensor warns because a readonly mmap would expose storage that
    # PyTorch can write to. Copy this array, keeping ordinary writable arrays
    # zero-copy and preserving tensor inputs' devices unless alignment is needed.
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = np.array(values, copy=True)
    return torch.as_tensor(values, device=device)


def regression_metrics(prediction, target) -> dict[str, float]:
    """Return scalar field errors, giving every component equal weight."""
    prediction = _metric_tensor(prediction)
    target = _metric_tensor(target, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim < 2 or prediction.numel() == 0:
        raise ValueError("metrics require a nonempty batch with a sample dimension")
    if torch.is_complex(prediction) or torch.is_complex(target):
        raise ValueError("metrics require real and imaginary values in separate channels")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")
    with torch.no_grad():
        difference = prediction.to(torch.float64) - target.to(torch.float64)
        squared = difference.square()
        absolute = difference.abs()
        result = {
            "mse": squared.mean().item(),
            "mae": absolute.mean().item(),
            "l1_sum": absolute.sum().item(),
            "max_sample_mse": squared.reshape(len(prediction), -1).mean(1).max().item(),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("errors are too large to produce finite metrics")
        return result
