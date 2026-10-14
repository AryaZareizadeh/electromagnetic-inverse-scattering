"""Portable forward-model training, prediction, and batched evaluation."""

import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn

from .checkpoints import load_checkpoint, save_checkpoint
from .metrics import regression_metrics
from .model import DConvNet, resolve_device


def _validate_domains(domains, *, allow_empty=False):
    if np.ndim(domains) != 4 or tuple(domains.shape[1:]) != (2, 28, 28):
        raise ValueError("domains must have shape (N, 2, 28, 28)")
    if np.iscomplexobj(domains):
        raise ValueError("domains must use two real channels")
    if not allow_empty and len(domains) == 0:
        raise ValueError("domains cannot be empty")


def _validate_pair(domains, fields):
    _validate_domains(domains)
    if tuple(fields.shape) != (len(domains), 2, 20, 20):
        raise ValueError("fields must have shape (N, 2, 20, 20) matching domains")
    if np.iscomplexobj(fields):
        raise ValueError("fields must use two real channels")


def _validate_batch_size(batch_size):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


def _tensor_batch(array, start, stop, device):
    # Copy only this batch. This supports read-only NumPy mmap arrays and avoids
    # retaining entire datasets on CUDA as the notebooks did.
    batch = np.array(array[start:stop], dtype=np.float32, copy=True)
    return torch.from_numpy(batch).to(device)


def predict_fields(model, domains, *, batch_size=100, device="cpu") -> np.ndarray:
    """Predict fields in supplied sample order, including a partial last batch."""
    _validate_domains(domains, allow_empty=True)
    _validate_batch_size(batch_size)
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    result = np.empty((len(domains), 2, 20, 20), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(domains), batch_size):
            stop = min(start + batch_size, len(domains))
            prediction = model(_tensor_batch(domains, start, stop, target_device))
            result[start:stop] = prediction.reshape(stop - start, 2, 20, 20).cpu().numpy()
    return result


def evaluate_forward(model, domains, fields, *, batch_size=100, device="cpu") -> dict:
    """Evaluate every field component without retaining all predictions."""
    _validate_pair(domains, fields)
    _validate_batch_size(batch_size)
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    squared_sum = absolute_sum = maximum = 0.0
    components = 0
    with torch.inference_mode():
        for start in range(0, len(domains), batch_size):
            stop = min(start + batch_size, len(domains))
            prediction = model(_tensor_batch(domains, start, stop, target_device))
            target = _tensor_batch(fields, start, stop, target_device).reshape(stop - start, -1)
            scores = regression_metrics(prediction, target)
            count = target.numel()
            squared_sum += scores["mse"] * count
            absolute_sum += scores["l1_sum"]
            maximum = max(maximum, scores["max_sample_mse"])
            components += count
    return {
        "mse": squared_sum / components,
        "mae": absolute_sum / components,
        "l1_sum": absolute_sum,
        "max_sample_mse": maximum,
    }


def _save_history(path, payload):
    descriptor, temporary_name = tempfile.mkstemp(prefix=".esdl-", suffix=".json", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_forward_model(
    train_domains, train_fields, test_domains, test_fields, output_dir, *,
    epochs=400, batch_size=100, learning_rate=1e-6, device="auto", seed=0,
    resume=None, full_batch=False, resume_learning_rate=None,
) -> dict:
    """Train additional epochs and save one final checkpoint and JSON history.

    Mini-batch training keeps the original sample order. ``full_batch=True``
    preserves the full-data experiment's one optimizer step per epoch; use its
    historical learning rate 5e-5 explicitly. Test metrics are observational;
    they do not select weights. Resume restores Adam's saved learning rate unless
    ``resume_learning_rate`` explicitly overrides it. Legacy checkpoints contain
    no epoch counter, so their new history begins at epoch 1.
    """
    _validate_pair(train_domains, train_fields)
    _validate_pair(test_domains, test_fields)
    _validate_batch_size(batch_size)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if resume_learning_rate is not None:
        if resume is None:
            raise ValueError("resume_learning_rate requires a resume checkpoint")
        if not np.isfinite(resume_learning_rate) or resume_learning_rate <= 0:
            raise ValueError("resume_learning_rate must be positive and finite")
    directory = Path(output_dir).absolute()
    checkpoint_path = directory / "forward_checkpoint.pt"
    history_path = directory / "history.json"
    if checkpoint_path.exists() or history_path.exists():
        raise FileExistsError("training output already exists; choose a new output directory")
    target_device = resolve_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_epoch = 0
    if resume is None:
        with torch.device(target_device):
            model = DConvNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        checkpoint = load_checkpoint(resume)
        with torch.device("meta"):
            model = DConvNet()
        model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
        model.to(target_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if resume_learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = resume_learning_rate
        start_epoch = int(checkpoint.get("epoch", 0))
        del checkpoint

    effective_batch_size = len(train_domains) if full_batch else batch_size
    config = {
        "epochs": epochs, "batch_size": effective_batch_size,
        "evaluation_batch_size": batch_size,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "device": str(target_device), "seed": seed, "full_batch": bool(full_batch),
        "resume": str(Path(resume).absolute()) if resume is not None else None,
        "start_epoch": start_epoch,
    }
    loss_function = nn.MSELoss()
    history = []
    for offset in range(epochs):
        epoch = start_epoch + offset + 1
        model.train()
        squared_sum = 0.0
        components = 0
        for start in range(0, len(train_domains), effective_batch_size):
            stop = min(start + effective_batch_size, len(train_domains))
            domains = _tensor_batch(train_domains, start, stop, target_device)
            target = _tensor_batch(train_fields, start, stop, target_device).reshape(stop - start, -1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(domains)
            loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training loss at epoch {epoch}")
            squared_sum += loss.detach().item() * target.numel()
            components += target.numel()
            loss.backward()
            optimizer.step()
        scores = evaluate_forward(
            model, test_domains, test_fields, batch_size=batch_size, device=target_device,
        )
        row = {"epoch": epoch, "train_mse": squared_sum / components}
        row.update({f"test_{name}": value for name, value in scores.items()})
        history.append(row)
        print(f"Epoch {epoch}: train MSE {row['train_mse']:.10g}, test MSE {scores['mse']:.10g}")

    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, checkpoint_path, epoch=start_epoch + epochs, config=config)
    _save_history(history_path, {"config": config, "history": history})
    return {
        "history": history, "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path), "config": config,
    }
