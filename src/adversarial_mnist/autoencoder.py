"""Autoencoder purifier used by the Adversarial Firewall extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from adversarial_mnist.training import load_checkpoint, save_checkpoint


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for 28x28 MNIST images."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructed images with shape ``[batch, 1, 28, 28]``."""
        return self.decoder(self.encoder(x))


def autoencoder_checkpoint_path(checkpoints_dir: str | Path, seed: int) -> Path:
    """Return the checkpoint path for a seed-specific autoencoder."""
    return Path(checkpoints_dir) / f"autoencoder_seed{seed}.pt"


def reconstruction_loss(reconstructed: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return mean-squared reconstruction loss."""
    if reconstructed.shape != inputs.shape:
        raise ValueError("Reconstructed tensor and inputs must have matching shapes.")
    return F.mse_loss(reconstructed, inputs)


def save_autoencoder_checkpoint(
    path: str | Path,
    autoencoder: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: dict[str, Any],
) -> None:
    """Save an autoencoder checkpoint with metadata."""
    save_checkpoint(path, autoencoder, optimizer, metadata)


def load_autoencoder_checkpoint(
    path: str | Path,
    autoencoder: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load an autoencoder checkpoint and return metadata."""
    return load_checkpoint(path, autoencoder, optimizer=optimizer, map_location=map_location)
