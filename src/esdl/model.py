"""The original DConvNet forward surrogate, with compatible parameter names."""

import torch
from torch import nn


class DConvNet(nn.Module):
    """Map two-channel 28×28 permittivity domains to 800 field components.

    The unusually large linear layer is intentional: changing its size or adding
    activations would change the saved model. Channels and flattening follow the
    supplied arrays, with real field values followed by imaginary values.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 50, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=2, stride=1, padding="same", bias=True),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(14 * 14 * 100, 28 * 28 * 25, bias=True)
        self.fc2 = nn.Linear(28 * 28 * 25, 20 * 20 * 2, bias=True)

    def forward(self, domains):
        if domains.ndim != 4 or tuple(domains.shape[1:]) != (2, 28, 28):
            raise ValueError("domains must have shape (N, 2, 28, 28)")
        output = self.conv2(self.conv1(domains))
        output = output.reshape(output.shape[0], 14 * 14 * 100)
        return self.fc2(self.fc1(output))


# The notebook's historical class name is retained for callers and provenance.
AlexNet = DConvNet


def resolve_device(name="auto") -> torch.device:
    """Choose an available device; an explicit unavailable CUDA request fails."""
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device: {name}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is not available")
    return device
