"""The original two-dimensional method-of-moments forward solver.

The default formulation reproduces the supplied training data, including the
notebook's use of row permittivity in the interaction impedance. ``free-space``
changes that term to the configured free-space impedance; it produces different
data and is not compatible with the supplied trained parameters. The remaining
notebook equations and self-interaction constants are preserved in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import hankel2


@dataclass(frozen=True)
class ForwardConfig:
    """Forward geometry in meters, with independent wavelength and domain size."""

    wavelength: float = 1.0
    domain_length: float = 1.0
    receivers: int = 20
    transmitters: int = 20
    receiver_radius: float = 20.0
    amplitude: float = 1.0
    impedance: float = 376.7303
    formulation: Literal["legacy", "free-space"] = "legacy"

    def __post_init__(self) -> None:
        for name in ("receivers", "transmitters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wavelength", "domain_length", "receiver_radius", "impedance"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if not np.isscalar(self.amplitude) or not np.isfinite(self.amplitude):
            raise ValueError("amplitude must be finite")
        if self.formulation not in ("legacy", "free-space"):
            raise ValueError("formulation must be 'legacy' or 'free-space'")


def solve_scattered_field(
    permittivity: ArrayLike, config: ForwardConfig | None = None
) -> NDArray[np.complex128]:
    """Solve one raw square permittivity map, returning [receiver, transmitter].

    Image rows increase downward; columns increase toward positive x. Values
    equal to one represent background. Only cells with permittivity above one
    participate in the current system. Processed project domains must first be
    decoded to their material channel; this function accepts a two-dimensional
    map, not the project's two-channel network encoding.
    """
    config = config if config is not None else ForwardConfig()
    domain = np.asarray(permittivity)
    if domain.ndim != 2 or domain.shape[0] == 0 or domain.shape[0] != domain.shape[1]:
        raise ValueError("permittivity must be a nonempty square two-dimensional map")
    if np.iscomplexobj(domain):
        raise ValueError("permittivity must be real")
    domain = domain.astype(np.float64, copy=False)
    if not np.all(np.isfinite(domain)) or np.any(domain < 1):
        raise ValueError("permittivity must contain finite values greater than or equal to one")

    rows, columns = np.nonzero(domain > 1)
    eps = domain[rows, columns]
    if len(eps) == 0:
        return np.zeros((config.receivers, config.transmitters), dtype=np.complex128)

    spacing = config.domain_length / domain.shape[0]
    area = spacing**2
    wavenumber = 2 * np.pi / config.wavelength
    x = -config.domain_length / 2 + (columns + 0.5) * spacing
    y = config.domain_length / 2 - (rows + 0.5) * spacing
    distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    # Avoid evaluating the singular Hankel function at zero. The original
    # notebook's diagonal approximation is substituted before matrix assembly.
    np.fill_diagonal(distances, 1.0)
    kernel = hankel2(0, wavenumber * distances)
    diagonal_kernel = 1 - 2j / np.pi * np.log(wavenumber * 1.7180724 * area / 4 / np.exp(1))
    np.fill_diagonal(kernel, diagonal_kernel)

    interaction_impedance = eps[:, None] if config.formulation == "legacy" else config.impedance
    matrix = wavenumber * interaction_impedance * area / 4 * kernel
    matrix[np.diag_indices(len(eps))] += -1j * config.impedance / (wavenumber * (eps - 1))

    transmitter_angles = np.arange(config.transmitters) * (2 * np.pi / config.transmitters)
    incident = config.amplitude * np.exp(
        1j * wavenumber * (
            x[:, None] * np.cos(transmitter_angles)
            + y[:, None] * np.sin(transmitter_angles)
        )
    )
    currents = np.linalg.solve(matrix, incident)

    receiver_angles = np.arange(config.receivers) * (2 * np.pi / config.receivers)
    xr = config.receiver_radius * np.cos(receiver_angles)
    yr = config.receiver_radius * np.sin(receiver_angles)
    receiver_distances = np.hypot(xr[:, None] - x, yr[:, None] - y)
    if np.any(receiver_distances == 0):
        raise ValueError("a receiver coincides with an active cell center")
    receiver_kernel = -wavenumber * config.impedance * area / 4 * hankel2(
        0, wavenumber * receiver_distances
    )
    return receiver_kernel @ currents


def simulate_fields(
    domains: ArrayLike, config: ForwardConfig | None = None
) -> NDArray[np.complex128]:
    """Solve raw [sample, row, column] maps without plotting or writing files."""
    config = config if config is not None else ForwardConfig()
    domains = np.asarray(domains)
    if domains.ndim != 3 or domains.shape[1] == 0 or domains.shape[1] != domains.shape[2]:
        raise ValueError("domains must have shape (samples, height, width) with nonempty square maps")
    fields = np.empty((len(domains), config.receivers, config.transmitters), dtype=np.complex128)
    for index, domain in enumerate(domains):
        fields[index] = solve_scattered_field(domain, config)
    return fields
