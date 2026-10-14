"""Checkpoint compatibility without allocating the 1.6 GB model."""

import pytest
import torch
from torch import nn


def test_model_matches_the_supplied_checkpoint_without_allocating_storage():
    from esdl.model import AlexNet, DConvNet

    with torch.device("meta"):
        model = DConvNet()
        prediction = model(torch.empty(3, 2, 28, 28))
    assert AlexNet is DConvNet
    assert prediction.shape == (3, 800)
    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == {
        "conv1.0.weight": (50, 2, 2, 2),
        "conv1.0.bias": (50,),
        "conv2.0.weight": (100, 50, 2, 2),
        "conv2.0.bias": (100,),
        "fc1.weight": (19600, 19600),
        "fc1.bias": (19600,),
        "fc2.weight": (800, 19600),
        "fc2.bias": (800,),
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == 399_880_950


def test_negative_linear_output_survives_the_forward_model():
    from esdl.model import DConvNet

    # Replace only the enormous linear weights, keeping the real forward method.
    with torch.device("meta"):
        model = DConvNet()
    model.conv1.to_empty(device="cpu")
    model.conv2.to_empty(device="cpu")
    model.fc1 = nn.Linear(19600, 1)
    model.fc2 = nn.Linear(1, 800)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc1.bias.fill_(-2)
        model.fc2.weight.fill_(1)
    assert torch.equal(model(torch.ones(1, 2, 28, 28)), torch.full((1, 800), -2.0))


def test_checkpoint_round_trip_keeps_optimizer_and_metadata(tmp_path):
    from esdl.checkpoints import save_checkpoint

    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path = tmp_path / "weights.pt"
    save_checkpoint(model, optimizer, path, epoch=3)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert payload["epoch"] == 3
    assert payload["optimizer"]["param_groups"][0]["lr"] == 1e-7
    assert torch.equal(payload["state_dict"]["weight"], model.weight)
    assert list(tmp_path.iterdir()) == [path]


def test_loader_assigns_saved_parameters_to_meta_model(tmp_path, monkeypatch):
    import esdl.checkpoints as checkpoints

    # A real tiny checkpoint exercises mmap/assignment without allocating 1.6 GB.
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, values):
            return self.fc(values)

    source = TinyModel()
    with torch.no_grad():
        source.fc.weight.fill_(3)
        source.fc.bias.fill_(-1)
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": source.state_dict()}, path)
    monkeypatch.setattr(checkpoints, "DConvNet", TinyModel)
    loaded = checkpoints.load_forward_model(path)
    assert loaded.fc.weight.device.type == "cpu"
    assert not loaded.training
    assert torch.equal(loaded(torch.ones(1, 2)), torch.full((1, 2), 5.0))


def test_tensor_loader_rejects_checkpoint_dictionaries(tmp_path):
    from esdl.checkpoints import load_tensor

    path = tmp_path / "tensor.pt"
    torch.save(torch.tensor([[1.0, 2.0]]), path)
    assert torch.equal(load_tensor(path), torch.tensor([[1.0, 2.0]]))
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="tensor"):
        load_tensor(path)


def test_resolve_device_does_not_silently_accept_unavailable_cuda(monkeypatch):
    from esdl.model import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")
