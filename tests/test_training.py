"""Exercise batching and resume with a small real trainable surrogate."""

import numpy as np
import pytest
import torch
from torch import nn


class TinyForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, domains):
        values = domains[:, 1, 0, 0] * self.scale
        return values[:, None].expand(-1, 800)


def sample_domains():
    result = np.zeros((3, 2, 28, 28), dtype=np.float64)
    result[:, 1, 0, 0] = [1, 2, 3]
    return result


def test_prediction_keeps_order_and_tail_in_channel_first_fields():
    from esdl.training import predict_fields

    model = TinyForward()
    fields = predict_fields(model, sample_domains(), batch_size=2)
    assert fields.shape == (3, 2, 20, 20)
    assert fields.dtype == np.float32
    np.testing.assert_array_equal(fields[:, 0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(fields[:, 1, -1, -1], [1, 2, 3])


def test_evaluation_weights_a_smaller_last_batch_by_its_sample_count():
    from esdl.training import evaluate_forward

    fields = np.zeros((3, 2, 20, 20))
    result = evaluate_forward(TinyForward(), sample_domains(), fields, batch_size=2)
    assert result["mse"] == pytest.approx(14 / 3)
    assert result["mae"] == 2
    assert result["l1_sum"] == 4800
    assert result["max_sample_mse"] == 9


def test_training_includes_tail_batch_and_saves_resumable_checkpoint(tmp_path, monkeypatch):
    import esdl.training as training

    monkeypatch.setattr(training, "DConvNet", TinyForward)
    domains = sample_domains()
    fields = np.zeros((3, 2, 20, 20))
    result = training.train_forward_model(
        domains, fields, domains, fields, tmp_path, epochs=1, batch_size=2,
        device="cpu", learning_rate=1e-3,
    )
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True, mmap=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["optimizer"]["state"][0]["step"].item() == 2
    assert checkpoint["state_dict"]["scale"].item() < 1
    assert len(result["history"]) == 1
    assert result["history"][0]["train_mse"] > 0
    assert result["history"][0]["test_mse"] > 0
    assert (tmp_path / "history.json").is_file()


def test_full_batch_training_performs_one_step_per_epoch(tmp_path, monkeypatch):
    import esdl.training as training

    monkeypatch.setattr(training, "DConvNet", TinyForward)
    domains = sample_domains()
    fields = np.zeros((3, 2, 20, 20))
    result = training.train_forward_model(
        domains, fields, domains, fields, tmp_path, epochs=1, batch_size=2,
        device="cpu", full_batch=True,
    )
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True, mmap=True)
    assert checkpoint["optimizer"]["state"][0]["step"].item() == 1


def test_resume_preserves_saved_learning_rate_and_continues_epoch_numbers(tmp_path, monkeypatch):
    import esdl.training as training
    from esdl.checkpoints import save_checkpoint

    monkeypatch.setattr(training, "DConvNet", TinyForward)
    model = TinyForward()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    path = tmp_path / "resume.pt"
    save_checkpoint(model, optimizer, path, epoch=7)
    domains = sample_domains()
    fields = np.zeros((3, 2, 20, 20))
    result = training.train_forward_model(
        domains, fields, domains, fields, tmp_path / "continued", epochs=1,
        batch_size=2, device="cpu", resume=path, learning_rate=0.1,
    )
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True, mmap=True)
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 1e-7
    assert checkpoint["epoch"] == 8


def test_empty_training_data_fails_before_allocating_the_large_model(tmp_path):
    from esdl.training import train_forward_model

    with pytest.raises(ValueError, match="empty"):
        train_forward_model(
            np.empty((0, 2, 28, 28)), np.empty((0, 2, 20, 20)),
            sample_domains(), np.zeros((3, 2, 20, 20)), tmp_path, epochs=1,
        )
