import numpy as np
import pytest

from esdl.data import encode_domains, encode_fields, load_split, validate_arrays


def test_domain_encoding_keeps_zero_then_permittivity():
    raw = np.arange(2 * 28 * 28).reshape(2, 28, 28)
    encoded = encode_domains(raw)
    np.testing.assert_array_equal(encoded[:, 0], 0)
    np.testing.assert_array_equal(encoded[:, 1], raw)


def test_field_encoding_preserves_flatten_order():
    field = np.arange(400).reshape(1, 20, 20) + 1j * 999
    encoded = encode_fields(field)
    np.testing.assert_array_equal(encoded.reshape(1, -1)[0, :400], field.real.ravel())
    np.testing.assert_array_equal(encoded.reshape(1, -1)[0, 400:], 999)


def test_load_split_slices_pairs_and_labels(tmp_path):
    domains = np.ones((3, 2, 28, 28))
    fields = np.ones((3, 2, 20, 20))
    np.save(tmp_path / "Domains.npy", domains)
    np.save(tmp_path / "Field_Data.npy", fields)
    np.save(tmp_path / "Labels.npy", [3, 2, 1])
    data = load_split(tmp_path, "train", limit=2)
    assert data.domains.shape[0] == data.fields.shape[0] == 2
    np.testing.assert_array_equal(data.labels, [3, 2])


def test_bad_shapes_and_nonfinite_data_are_rejected():
    domains = np.ones((2, 2, 28, 28))
    fields = np.ones((1, 2, 20, 20))
    with pytest.raises(ValueError, match="sample"):
        validate_arrays(domains, fields)
    fields = np.full((2, 2, 20, 20), np.nan)
    with pytest.raises(ValueError, match="finite"):
        validate_arrays(domains, fields)


def test_missing_labels_are_optional(tmp_path):
    np.save(tmp_path / "Domains_cy.npy", np.ones((1, 2, 28, 28)))
    np.save(tmp_path / "Field_Data_cy.npy", np.ones((1, 2, 20, 20)))
    assert load_split(tmp_path, "cylinders").labels is None


def test_empty_and_invalid_limits_are_rejected(tmp_path):
    np.save(tmp_path / "Domains.npy", np.ones((1, 2, 28, 28)))
    np.save(tmp_path / "Field_Data.npy", np.ones((1, 2, 20, 20)))
    with pytest.raises(ValueError, match="positive"):
        load_split(tmp_path, "train", limit=0)
    with pytest.raises(ValueError, match="shape"):
        encode_domains(np.ones((28, 28)))
