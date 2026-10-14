"""CPU memory-mapped artifact loading and atomic checkpoint saving."""

import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import torch

from .model import DConvNet, resolve_device


def load_checkpoint(path):
    """Read a state dictionary/checkpoint without materializing its storages.

    The project's artifacts use PyTorch's ZIP serialization. Restrict unpickling
    to weights and map all original CUDA storages onto CPU. The mmap is private:
    subsequent tensor updates do not write back to the source artifact.
    """
    payload = torch.load(
        Path(path).absolute(), map_location="cpu", weights_only=True, mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("expected a checkpoint dictionary containing model weights")
    if "state_dict" in payload:
        state = payload["state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint state_dict must be a dictionary")
        return payload
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return {"state_dict": payload}
    raise ValueError("checkpoint must contain state_dict model weights")


def load_forward_model(path, device="cpu"):
    """Load supplied DConvNet weights without a second 1.6 GB allocation."""
    target_device = resolve_device(device)
    checkpoint = load_checkpoint(path)
    with torch.device("meta"):
        model = DConvNet()
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    model.to(target_device)
    model.eval()
    return model


def load_tensor(path) -> torch.Tensor:
    """Load a saved tensor onto CPU using private memory-mapped storage."""
    tensor = torch.load(
        Path(path).absolute(), map_location="cpu", weights_only=True, mmap=True,
    )
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("expected a saved tensor")
    return tensor


def save_checkpoint(model, optimizer, path, **metadata):
    """Atomically save parameters, optional optimizer state, and run metadata."""
    if "state_dict" in metadata or "optimizer" in metadata:
        raise ValueError("metadata cannot replace state_dict or optimizer")
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": model.state_dict(), **metadata}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".esdl-", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
