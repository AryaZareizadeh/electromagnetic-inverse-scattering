"""Numerical regression checks against the original notebook equations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import hankel2


def notebook_reference(permittivity, config):
    """Literal per-transmitter loops, independent of production helpers."""
    width = len(permittivity)
    spacing = config.domain_length / width
    area = spacing**2
    wavenumber = 2 * np.pi / config.wavelength
    positions = []
    values = []
    for row in range(width):
        for column in range(width):
            if permittivity[row, column] > 1:
                positions.append(
                    (
                        -config.domain_length / 2 + (column + 0.5) * spacing,
                        config.domain_length / 2 - (row + 0.5) * spacing,
                    )
                )
                values.append(permittivity[row, column])
    fields = np.zeros((config.receivers, config.transmitters), dtype=complex)
    for transmitter in range(config.transmitters):
        angle = 2 * np.pi * transmitter / config.transmitters
        matrix = np.zeros((len(values), len(values)), dtype=complex)
        incident = np.zeros(len(values), dtype=complex)
        for m, ((xm, ym), eps) in enumerate(zip(positions, values)):
            incident[m] = config.amplitude * np.exp(
                1j * wavenumber * (xm * np.cos(angle) + ym * np.sin(angle))
            )
            impedance = eps if config.formulation == "legacy" else config.impedance
            for n, (xn, yn) in enumerate(positions):
                if m == n:
                    matrix[m, n] = wavenumber * impedance * area / 4 * (
                        1
                        - 2j / np.pi
                        * np.log(wavenumber * 1.7180724 * area / 4 / np.exp(1))
                    )
                else:
                    distance = np.sqrt((xm - xn) ** 2 + (ym - yn) ** 2)
                    matrix[m, n] = (
                        wavenumber * impedance * area / 4
                        * hankel2(0, wavenumber * distance)
                    )
            matrix[m, m] += -1j * config.impedance / (wavenumber * (eps - 1))
        if not values:
            continue
        currents = np.linalg.inv(matrix) @ incident
        for receiver in range(config.receivers):
            angle = 2 * np.pi * receiver / config.receivers
            xr = config.receiver_radius * np.cos(angle)
            yr = config.receiver_radius * np.sin(angle)
            for n, (xn, yn) in enumerate(positions):
                distance = np.sqrt((xr - xn) ** 2 + (yr - yn) ** 2)
                fields[receiver, transmitter] += (
                    -wavenumber * config.impedance * currents[n] * area / 4
                    * hankel2(0, wavenumber * distance)
                )
    return fields


@pytest.mark.parametrize("formulation", ["legacy", "free-space"])
def test_vectorized_solver_matches_literal_notebook(formulation):
    from esdl.forward import ForwardConfig, solve_scattered_field

    # Asymmetric locations and distinct materials catch x/y and row scaling errors.
    domain = np.array([[2.0, 1.0, 1.1], [1.0, 1.3, 1.0], [1.0, 1.0, 1.8]])
    config = ForwardConfig(
        wavelength=0.8, domain_length=0.6, receivers=3, transmitters=5,
        receiver_radius=8.0, amplitude=1.4, formulation=formulation,
    )
    result = solve_scattered_field(domain, config)
    assert result.shape == (3, 5)
    np.testing.assert_allclose(result, notebook_reference(domain, config), rtol=1e-12, atol=1e-14)


def test_background_produces_zero_scattered_field():
    from esdl.forward import ForwardConfig, solve_scattered_field

    config = ForwardConfig(receivers=4, transmitters=2)
    result = solve_scattered_field(np.ones((3, 3)), config)
    assert result.shape == (4, 2)
    assert np.iscomplexobj(result)
    assert np.count_nonzero(result) == 0


def test_default_formulation_preserves_legacy_material_impedance():
    from esdl.forward import ForwardConfig, solve_scattered_field

    domain = np.array([[2.0, 1.0], [1.0, 1.5]])
    config = ForwardConfig(receivers=3, transmitters=4)
    expected = notebook_reference(domain, config)
    np.testing.assert_allclose(solve_scattered_field(domain, config), expected, rtol=1e-12)
    corrected = solve_scattered_field(domain, replace(config, formulation="free-space"))
    assert np.linalg.norm(corrected - expected) / np.linalg.norm(expected) > 0.01


def test_batch_solver_preserves_sample_and_receiver_transmitter_order():
    from esdl.forward import ForwardConfig, simulate_fields

    domains = np.array([[[2.0, 1.0], [1.0, 1.5]], [[1.0, 1.0], [1.8, 1.0]]])
    config = ForwardConfig(receivers=3, transmitters=2)
    result = simulate_fields(domains, config)
    assert result.shape == (2, 3, 2)
    for i in range(len(domains)):
        np.testing.assert_allclose(result[i], notebook_reference(domains[i], config), rtol=1e-12)


def test_empty_batch_has_known_field_dimensions():
    from esdl.forward import ForwardConfig, simulate_fields

    fields = simulate_fields(np.empty((0, 28, 28)), ForwardConfig(receivers=3, transmitters=2))
    assert fields.shape == (0, 3, 2)
    assert np.iscomplexobj(fields)


@pytest.mark.parametrize("domain", [
    np.ones((2, 3)), np.ones((2, 2, 2)), np.empty((0, 0)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[0.5]]),
    np.array([[1.0 + 0.2j]]),
])
def test_invalid_domain_is_rejected(domain):
    from esdl.forward import solve_scattered_field

    with pytest.raises(ValueError):
        solve_scattered_field(domain)


@pytest.mark.parametrize("kwargs", [
    {"receivers": 0}, {"transmitters": 1.5}, {"wavelength": 0},
    {"domain_length": -1}, {"receiver_radius": np.nan}, {"impedance": 0},
    {"amplitude": np.inf}, {"formulation": "unknown"},
])
def test_invalid_geometry_is_rejected(kwargs):
    from esdl.forward import ForwardConfig

    with pytest.raises(ValueError):
        ForwardConfig(**kwargs)
