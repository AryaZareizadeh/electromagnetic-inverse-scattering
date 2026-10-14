"""Regression metrics use their arguments and count every sample."""

import warnings

import numpy as np
import pytest
import torch


def test_regression_metrics_report_hand_calculated_errors():
    from esdl.metrics import regression_metrics

    prediction = torch.tensor([[1.0, -1.0], [3.0, -3.0]])
    target = torch.zeros_like(prediction)
    assert regression_metrics(prediction, target) == {
        "mse": 5.0, "mae": 2.0, "l1_sum": 8.0, "max_sample_mse": 9.0,
    }
    assert regression_metrics(torch.zeros_like(target), target)["l1_sum"] == 0


def test_regression_metrics_reject_broadcasting():
    from esdl.metrics import regression_metrics

    with pytest.raises(ValueError, match="shape"):
        regression_metrics(torch.ones(2, 2), torch.ones(2, 1))


def test_readonly_memory_mapped_arrays_produce_metrics_without_warnings(tmp_path):
    from esdl.metrics import regression_metrics

    prediction_path = tmp_path / "prediction.npy"
    target_path = tmp_path / "target.npy"
    np.save(prediction_path, [[1.0, -1.0], [3.0, -3.0]])
    np.save(target_path, np.zeros((2, 2)))
    prediction = np.load(prediction_path, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = regression_metrics(prediction, target)
    assert not emitted
    assert result == {"mse": 5.0, "mae": 2.0, "l1_sum": 8.0, "max_sample_mse": 9.0}
    assert not prediction.flags.writeable
    assert not target.flags.writeable
    np.testing.assert_array_equal(np.load(prediction_path), [[1.0, -1.0], [3.0, -3.0]])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_side", ["prediction", "target"])
def test_regression_metrics_reject_nonfinite_inputs(invalid, invalid_side):
    from esdl.metrics import regression_metrics

    values = {"prediction": torch.zeros(1, 2), "target": torch.zeros(1, 2)}
    values[invalid_side][0, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(**values)


def test_finite_float32_errors_are_accumulated_without_float32_overflow():
    from esdl.metrics import regression_metrics

    result = regression_metrics(torch.tensor([[3e38]]), torch.tensor([[-3e38]]))
    assert result["mse"] == pytest.approx(3.6e77)


def test_errors_that_overflow_metric_storage_fail_instead_of_emitting_infinity():
    from esdl.metrics import regression_metrics

    with pytest.raises(ValueError, match="finite"):
        regression_metrics(torch.tensor([[1e200]], dtype=torch.float64), torch.zeros(1, 1))
