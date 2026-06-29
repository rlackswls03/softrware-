"""Tests for the MNIST autoencoder purifier."""

from __future__ import annotations

import torch

from adversarial_mnist.autoencoder import ConvAutoencoder, reconstruction_loss


def test_autoencoder_output_shape_and_range() -> None:
    model = ConvAutoencoder()
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        reconstructed = model(x)
    assert reconstructed.shape == x.shape
    assert float(reconstructed.min()) >= 0.0
    assert float(reconstructed.max()) <= 1.0


def test_reconstruction_loss_is_scalar() -> None:
    model = ConvAutoencoder()
    x = torch.rand(4, 1, 28, 28)
    reconstructed = model(x)
    loss = reconstruction_loss(reconstructed, x)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0
