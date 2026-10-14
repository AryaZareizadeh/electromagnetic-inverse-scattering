"""Offline supervised descent fitting and online inverse reconstruction.

The surrogate predicts fields from domains. Residuals are predicted minus
observed fields, so the fitted descent matrix is added to each current guess.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .model import resolve_device



def _require_finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f'{name} must contain only finite values.')


def load_descent_matrices(path: str | Path) -> Tensor:
    """Read the logical trained iteration count without copying padded storage.

    The original artifact is a 342-iteration view of an 800-iteration storage.
    Memory mapping preserves that view and keeps the matrices on the CPU.
    """
    from .checkpoints import load_tensor

    matrices = load_tensor(path)
    if matrices.ndim != 3 or matrices.shape[0] == 0 or tuple(matrices.shape[1:]) != (800, 1568):
        raise ValueError('Descent matrices must have shape (iterations, 800, 1568).')
    return matrices.to(device='cpu', dtype=torch.float32)


def load_initial_guess(path: str | Path, count: int, device: str | torch.device = 'cpu') -> Tensor:
    """Load supplied guesses, preserving both channels exactly.

    A uniform set of guesses can be broadcast to a larger dataset. A stored
    nonuniform set can only be sliced, because extra guesses would be ambiguous.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError('Initial guess count must be a positive integer.')
    from .checkpoints import load_tensor

    initial = load_tensor(path)
    if initial.ndim != 4 or initial.shape[0] == 0 or tuple(initial.shape[1:]) != (2, 28, 28):
        raise ValueError('Initial guesses must have shape (samples, 2, 28, 28).')
    _require_finite(initial, 'Initial guesses')
    if count <= len(initial):
        selected = initial[:count]
    else:
        if not torch.equal(initial, initial[:1].expand_as(initial)):
            raise ValueError('Cannot increase the count of a nonuniform set of initial guesses.')
        selected = initial[:1].expand(count, -1, -1, -1)
    return selected.to(device=resolve_device(device), dtype=torch.float32).clone()


def fit_descent_matrix(residual: Tensor, displacement: Tensor, alpha: float = 0.5) -> Tensor:
    """Fit ``residual @ K`` to ``displacement`` with Tikhonov regularization.

    This reduced SVD expression is algebraically identical to the notebook's
    ``V @ inverse(diag(S**2) + alpha**2 I) @ diag(S) @ U.T @ displacement``.
    It also works when there are fewer training samples than field features.
    """
    if residual.ndim != 2 or displacement.ndim != 2 or residual.shape[0] != displacement.shape[0]:
        raise ValueError('Residual and displacement must be matrices with matching sample counts.')
    if min(residual.shape) == 0 or displacement.shape[1] == 0:
        raise ValueError('Residual and displacement matrices must be nonempty.')
    if not residual.is_floating_point() or not displacement.is_floating_point():
        raise ValueError('Residual and displacement must use floating point values.')
    if residual.device != displacement.device or residual.dtype != displacement.dtype:
        raise ValueError('Residual and displacement must share a device and floating point dtype.')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Regularization alpha must be finite and greater than zero.')
    _require_finite(residual, 'Field residual')
    _require_finite(displacement, 'Domain displacement')
    u, singular_values, vh = torch.linalg.svd(residual, full_matrices=False)
    factors = singular_values / (singular_values.square() + alpha**2)
    matrix = vh.mT @ (factors[:, None] * (u.mT @ displacement))
    _require_finite(matrix, 'Fitted descent matrix')
    return matrix


def _float_tensor(values: Tensor | np.ndarray, device: torch.device) -> Tensor:
    if isinstance(values, Tensor):
        if values.is_complex():
            raise ValueError('Encoded domains and fields must be real valued.')
        return values.detach().to(device=device, dtype=torch.float32)
    if np.iscomplexobj(values):
        raise ValueError('Encoded domains and fields must be real valued.')
    # Dataset arrays are read-only memory maps; copy before exposing them to torch.
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)


def _prepare_inputs(model: nn.Module, fields: Tensor | np.ndarray,
                    initial: Tensor | np.ndarray, batch_size: int,
                    device: str | torch.device) -> tuple[Tensor, Tensor]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('Prediction batch size must be a positive integer.')
    device = resolve_device(device)
    estimates = _float_tensor(initial, device).clone()
    observed = _float_tensor(fields, device)
    if estimates.ndim < 2 or estimates.shape[0] == 0 or observed.ndim < 2 or len(observed) != len(estimates):
        raise ValueError('Fields and initial guesses must be nonempty with matching sample counts.')
    if estimates[0].numel() == 0 or observed[0].numel() == 0:
        raise ValueError('Fields and initial guesses must have nonempty feature dimensions.')
    observed = observed.flatten(1)
    _require_finite(estimates, 'Initial guesses')
    _require_finite(observed, 'Observed fields')
    model.to(device)
    model.eval()
    return observed, estimates


def _predict_fields(model: nn.Module, estimates: Tensor, features: int, batch_size: int) -> Tensor:
    predictions = []
    for start in range(0, len(estimates), batch_size):
        batch = estimates[start:start + batch_size]
        output = model(batch)
        if not isinstance(output, Tensor) or output.ndim < 2 or len(output) != len(batch):
            raise ValueError('The surrogate must return a field tensor for each input sample.')
        output = output.flatten(1)
        if output.shape[1] != features:
            raise ValueError(f'The surrogate returns {output.shape[1]} field features; expected {features}.')
        _require_finite(output, 'The surrogate predictions')
        predictions.append(output)
    return torch.cat(predictions, dim=0)


@torch.inference_mode()
def train_descent(model: nn.Module, domains: Tensor | np.ndarray, fields: Tensor | np.ndarray,
                  initial: Tensor | np.ndarray, *,
                  iterations: int = 800, alpha: float = 0.5, tolerance: float = 0.02,
                  batch_size: int = 100, device: str | torch.device = 'cpu',
                  output_path: str | Path | None = None) -> tuple[Tensor, list[dict[str, float | int]]]:
    """Learn descent directions, retaining the iteration that triggers stopping.

    The stopping score preserves the original ``mean(sum(error**2, dim=0))``.
    Returned matrices are a compact CPU float32 stack, unlike the original
    padded artifact. Negative tolerance can disable the usual plateau cutoff.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError('Training iterations must be a positive integer.')
    if not math.isfinite(tolerance):
        raise ValueError('Stopping tolerance must be finite.')
    if domains.shape != initial.shape:
        raise ValueError('Training domains and initial guesses must have identical shapes.')
    observed, estimates = _prepare_inputs(model, fields, initial, batch_size, device)
    target = _float_tensor(domains, estimates.device).flatten(1)
    _require_finite(target, 'Training domains')
    estimate_shape = estimates.shape
    best_score = 99_999_999.0
    learned = []
    history = []
    for index in range(iterations):
        predictions = _predict_fields(model, estimates, observed.shape[1], batch_size)
        residual = predictions - observed
        _require_finite(residual, 'Field residual')
        displacement = target - estimates.flatten(1)
        matrix = fit_descent_matrix(residual, displacement, alpha)
        estimates = (estimates.flatten(1) + residual @ matrix).reshape(estimate_shape)
        _require_finite(estimates, 'Updated domain guesses')
        learned.append(matrix.detach().to(device='cpu', dtype=torch.float32))
        score = (estimates.flatten(1) - target).square().sum(dim=0).mean().item()
        if not math.isfinite(score):
            raise ValueError('The training score is not finite; inspect the model and field scale.')
        improvement = best_score - score
        history.append({'iteration': index, 'score': score, 'improvement': improvement})
        if improvement <= tolerance:
            break
        if score < best_score:
            best_score = score
    matrices = torch.stack(learned).contiguous()
    del learned
    if output_path is not None:
        path = Path(output_path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(matrices, path)
    return matrices, history


@torch.inference_mode()
def reconstruct(model: nn.Module, fields: Tensor | np.ndarray, initial: Tensor | np.ndarray,
                matrices: Tensor, *,
                iterations: int | None = None, batch_size: int = 100,
                device: str | torch.device = 'cpu') -> Tensor:
    """Apply learned directions to observed fields, returning domain-shaped guesses.

    The stored matrix sequence remains on the CPU. Only the active direction
    is transferred to the reconstruction device at each iteration.
    """
    observed, estimates = _prepare_inputs(model, fields, initial, batch_size, device)
    if matrices.ndim != 3 or matrices.shape[0] == 0:
        raise ValueError('Descent matrices must be a nonempty three-dimensional tensor.')
    expected = (observed.shape[1], estimates[0].numel())
    if tuple(matrices.shape[1:]) != expected:
        raise ValueError(f'Descent matrix feature shape must be {expected}, got {tuple(matrices.shape[1:])}.')
    count = len(matrices) if iterations is None else iterations
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(matrices):
        raise ValueError(f'Reconstruction iterations must be between zero and {len(matrices)}.')
    estimate_shape = estimates.shape
    for index in range(count):
        matrix = matrices[index].to(device=estimates.device, dtype=torch.float32)
        _require_finite(matrix, f'Descent matrix {index}')
        predictions = _predict_fields(model, estimates, observed.shape[1], batch_size)
        residual = predictions - observed
        _require_finite(residual, 'Field residual')
        estimates = (estimates.flatten(1) + residual @ matrix).reshape(estimate_shape)
        _require_finite(estimates, f'Updated domain guesses at iteration {index}')
    return estimates
