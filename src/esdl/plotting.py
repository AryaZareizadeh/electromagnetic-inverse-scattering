"""Optional static plots for evaluating a forward prediction or reconstruction."""

from pathlib import Path

import numpy as np


def save_comparison(path: str | Path, domain, measured_field, predicted_field,
                    reconstructed=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def limits(*arrays):
        return min(float(np.min(a)) for a in arrays), max(float(np.max(a)) for a in arrays)

    domain_limits = limits(domain[1], reconstructed[1]) if reconstructed is not None else limits(domain[1])
    real_limits = limits(measured_field[0], predicted_field[0])
    imaginary_limits = limits(measured_field[1], predicted_field[1])
    images = [(domain[1], "Relative permittivity", domain_limits),
              (measured_field[0], "Observed field (real)", real_limits),
              (predicted_field[0], "Predicted field (real)", real_limits),
              (measured_field[1], "Observed field (imaginary)", imaginary_limits),
              (predicted_field[1], "Predicted field (imaginary)", imaginary_limits)]
    if reconstructed is not None:
        images.insert(1, (reconstructed[1], "Reconstructed permittivity", domain_limits))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (image, title, (minimum, maximum)) in zip(axes.flat, images):
        picture = axis.imshow(image, cmap="viridis", interpolation="none", vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(picture, ax=axis, shrink=0.8)
    for axis in list(axes.flat)[len(images):]:
        axis.set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_error_histogram(path: str | Path, observed, predicted) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    errors = np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2,
                     axis=(1, 2, 3))
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist(errors, bins=min(30, max(1, len(errors))))
    axis.set_xlabel("Mean squared field error per sample")
    axis.set_ylabel("Sample count")
    fig.savefig(path, dpi=150)
    plt.close(fig)
