"""Array conventions recovered from the supplied research datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPLIT_FILES = {
    "train": ("Domains.npy", "Field_Data.npy", "Labels.npy"),
    "test": ("Domains_test.npy", "Field_Data_test.npy", "Labels_test.npy"),
    "cylinders": ("Domains_cy.npy", "Field_Data_cy.npy", None),
}


@dataclass(frozen=True)
class ScatteringData:
    domains: np.ndarray
    fields: np.ndarray
    labels: np.ndarray | None = None


def encode_domains(permittivity: np.ndarray) -> np.ndarray:
    """Encode real maps as [zero, relative permittivity], never [real, imag]."""
    values = np.asarray(permittivity)
    if values.ndim != 3 or values.shape[1:] != (28, 28) or np.iscomplexobj(values):
        raise ValueError("Raw domains must have real shape (N, 28, 28)")
    if not np.isfinite(values).all():
        raise ValueError("Domains must contain finite values")
    return np.stack((np.zeros_like(values), values), axis=1)


def encode_fields(fields: np.ndarray) -> np.ndarray:
    """Encode complex [sample, receiver, transmitter] fields as [real, imag]."""
    values = np.asarray(fields)
    if values.ndim != 3 or values.shape[1:] != (20, 20):
        raise ValueError("Raw fields must have shape (N, 20, 20)")
    if not np.isfinite(values).all():
        raise ValueError("Fields must contain finite values")
    return np.stack((values.real, values.imag), axis=1)


def validate_arrays(domains: np.ndarray, fields: np.ndarray) -> None:
    if domains.ndim != 4 or domains.shape[1:] != (2, 28, 28):
        raise ValueError("Domain shape must be (N, 2, 28, 28)")
    if fields.ndim != 4 or fields.shape[1:] != (2, 20, 20):
        raise ValueError("Field shape must be (N, 2, 20, 20)")
    if len(domains) != len(fields) or len(domains) == 0:
        raise ValueError("Domain and field sample counts must match and be nonzero")
    if np.iscomplexobj(domains) or np.iscomplexobj(fields):
        raise ValueError("Encoded arrays must be real; use the channel conversion functions")
    if not np.isfinite(domains).all() or not np.isfinite(fields).all():
        raise ValueError("Arrays must contain finite values")


def load_split(directory: str | Path, split: str, limit: int | None = None) -> ScatteringData:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choose {', '.join(SPLIT_FILES)}")
    if limit is not None and limit < 1:
        raise ValueError("Sample limit must be positive")
    directory = Path(directory)
    domain_name, field_name, label_name = SPLIT_FILES[split]
    domains = np.load(directory / domain_name, mmap_mode="r", allow_pickle=False)
    fields = np.load(directory / field_name, mmap_mode="r", allow_pickle=False)
    labels = None
    if label_name and (directory / label_name).is_file():
        labels = np.load(directory / label_name, mmap_mode="r", allow_pickle=False)
        if labels.ndim != 1 or len(labels) != len(domains):
            raise ValueError("Labels must have one entry for each domain sample")
    # Validate complete pair lengths before taking a potentially masking subset.
    if len(domains) != len(fields):
        raise ValueError("Domain and field sample counts must match")
    if limit is not None:
        domains, fields = domains[:limit], fields[:limit]
        labels = labels[:limit] if labels is not None else None
    validate_arrays(domains, fields)
    return ScatteringData(domains, fields, labels)
