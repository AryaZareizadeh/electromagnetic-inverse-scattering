"""Numerical regression checks for the extracted supervised descent workflow."""

import pytest
import torch
from torch import nn


def test_svd_direction_matches_notebook_and_regularized_normal_equations():
    from esdl.descent import fit_descent_matrix

    generator = torch.Generator().manual_seed(12)
    residual = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    displacement = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    alpha = 0.5
    u, s, v = torch.svd(residual)
    notebook = v @ torch.inverse(torch.diag(s.square()) + alpha**2 * torch.eye(4, dtype=s.dtype)) @ torch.diag(s) @ u.T @ displacement
    expected = torch.linalg.solve(residual.T @ residual + alpha**2 * torch.eye(4, dtype=residual.dtype), residual.T @ displacement)

    actual = fit_descent_matrix(residual, displacement, alpha)

    torch.testing.assert_close(actual, notebook)
    torch.testing.assert_close(actual, expected)


def test_svd_direction_handles_rank_deficiency_and_rejects_nonfinite_inputs():
    from esdl.descent import fit_descent_matrix

    residual = torch.zeros(2, 5)
    displacement = torch.ones(2, 3)
    torch.testing.assert_close(fit_descent_matrix(residual, displacement), torch.zeros(5, 3))
    residual[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fit_descent_matrix(residual, displacement)


def test_training_keeps_stopping_iteration_and_saves_compact_tensor(tmp_path):
    from esdl.descent import train_descent

    domains = torch.tensor([[1., 3.], [2., 4.]]).reshape(2, 1, 1, 2)
    fields = domains.flatten(1)
    initial = torch.zeros_like(domains)
    path = tmp_path / 'directions.pt'

    matrices, history = train_descent(nn.Flatten(start_dim=1), domains, fields, initial, iterations=8, tolerance=1e9, batch_size=1, output_path=path)

    # A deliberately large threshold stops on the first comparison, after its update.
    assert matrices.shape == (1, 2, 2)
    assert len(history) == 1
    updated = initial.flatten(1) + (initial.flatten(1) - fields) @ matrices[0]
    expected_score = (updated - domains.flatten(1)).square().sum(dim=0).mean().item()
    assert history[0]['score'] == pytest.approx(expected_score)
    assert history[0]['improvement'] == pytest.approx(99_999_999.0 - expected_score)
    loaded = torch.load(path, weights_only=True)
    assert loaded.untyped_storage().nbytes() == loaded.numel() * loaded.element_size()
    torch.testing.assert_close(loaded, matrices)


def test_training_and_reconstruction_preserve_plus_residual_update_and_batching():
    from esdl.descent import reconstruct, train_descent

    domains = torch.tensor([[1., 2.], [2., 1.], [3., 1.]]).reshape(3, 1, 1, 2)
    fields = domains.flatten(1)
    initial = torch.zeros_like(domains)
    model = nn.Flatten(start_dim=1)
    matrices, history = train_descent(model, domains, fields, initial, iterations=3, tolerance=-1., batch_size=2)
    reconstructed = reconstruct(model, fields, initial, matrices, batch_size=2)
    assert len(history) == 3
    assert reconstructed.shape == initial.shape
    assert (reconstructed - domains).square().sum() < (initial - domains).square().sum()
    expected = initial.flatten(1)
    for matrix in matrices:
        expected = expected + (expected - fields) @ matrix
    torch.testing.assert_close(reconstructed.flatten(1), expected)
    partial = reconstruct(model, fields, initial, matrices, iterations=1, batch_size=1)
    torch.testing.assert_close(partial.flatten(1), initial.flatten(1) + (initial.flatten(1) - fields) @ matrices[0])


def test_nonfinite_surrogate_predictions_fail_with_actionable_message():
    from esdl.descent import reconstruct, train_descent

    class InvalidSurrogate(nn.Module):
        def forward(self, x):
            return torch.full((len(x), 2), float('inf'), device=x.device)

    domains = torch.ones(2, 1, 1, 2)
    fields = domains.flatten(1)
    initial = torch.zeros_like(domains)
    with pytest.raises(ValueError, match='surrogate.*finite'):
        train_descent(InvalidSurrogate(), domains, fields, initial, iterations=1)
    with pytest.raises(ValueError, match='surrogate.*finite'):
        reconstruct(InvalidSurrogate(), fields, initial, torch.zeros(1, 2, 2))


def test_initial_guess_preserves_two_channels_and_can_resize_uniform_artifact(tmp_path):
    from esdl.descent import load_initial_guess

    path = tmp_path / 'initial.pt'
    torch.save(torch.full((2, 2, 28, 28), 2.), path)
    initial = load_initial_guess(path, 5)
    assert initial.shape == (5, 2, 28, 28)
    assert initial.dtype == torch.float32
    assert torch.all(initial == 2.)
    torch.testing.assert_close(load_initial_guess(path, 1), initial[:1])
    with pytest.raises(ValueError, match='count'):
        load_initial_guess(path, 0)


def test_oversize_nonuniform_initial_guess_is_rejected(tmp_path):
    from esdl.descent import load_initial_guess

    path = tmp_path / 'initial.pt'
    tensor = torch.zeros(2, 2, 28, 28)
    tensor[1] = 1.
    torch.save(tensor, path)
    with pytest.raises(ValueError, match='uniform'):
        load_initial_guess(path, 3)


def test_descent_artifact_reads_logical_shape_and_maps_cpu(tmp_path):
    from esdl.descent import load_descent_matrices

    path = tmp_path / 'directions.pt'
    backing = torch.zeros(3, 800, 1568)
    torch.save(backing[:2], path)
    loaded = load_descent_matrices(path)
    assert loaded.shape == (2, 800, 1568)
    assert loaded.device.type == 'cpu'
    assert loaded.dtype == torch.float32
    assert loaded.untyped_storage().nbytes() == backing.untyped_storage().nbytes()


def test_descent_accepts_readonly_numpy_datasets_used_by_the_cli():
    import numpy as np
    from esdl.descent import reconstruct, train_descent

    domains = np.array([[1., 2.], [2., 1.]], dtype=np.float64).reshape(2, 1, 1, 2)
    fields = domains.reshape(2, 2)
    domains.setflags(write=False)
    fields.setflags(write=False)
    initial = torch.zeros(domains.shape)
    model = nn.Flatten(start_dim=1)
    matrices, _ = train_descent(model, domains, fields, initial, iterations=1)
    result = reconstruct(model, fields, initial, matrices)
    assert result.dtype == torch.float32
    assert (result - torch.tensor(domains.copy(), dtype=torch.float32)).square().sum() < initial.square().sum() + float(np.square(domains).sum())


def test_initial_guess_accepts_auto_device(tmp_path, monkeypatch):
    from esdl.descent import load_initial_guess

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    path = tmp_path / 'initial.pt'
    torch.save(torch.full((1, 2, 28, 28), 2.), path)
    assert load_initial_guess(path, 1, device='auto').device.type == 'cpu'
