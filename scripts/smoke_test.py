"""Smoke test without MNIST downloads or long training."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch

from adversarial_mnist.attacks import fgsm_attack, pgd_linf_attack
from adversarial_mnist.data import create_synthetic_dataloaders
from adversarial_mnist.evaluation import write_csv
from adversarial_mnist.models import create_model
from adversarial_mnist.training import train_fgsm_one_step, train_standard_one_step
from adversarial_mnist.utils import ensure_dir, set_reproducibility, setup_logging
from adversarial_mnist.visualization import plot_adversarial_examples, plot_robustness_curve

LOGGER = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    set_reproducibility(42)
    device = torch.device("cpu")
    loaders = create_synthetic_dataloaders(seed=42, batch_size=8)
    inputs, targets = next(iter(loaders.train))
    inputs = inputs.to(device)
    targets = targets.to(device)

    model = create_model("lenet").to(device)
    logits = model(inputs)
    if logits.shape != (inputs.shape[0], 10):
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

    x_fgsm = fgsm_attack(model, inputs, targets, epsilon=0.1)
    x_pgd = pgd_linf_attack(model, inputs, targets, epsilon=0.1, steps=3)
    if not (float(x_fgsm.min()) >= 0.0 and float(x_fgsm.max()) <= 1.0):
        raise RuntimeError("FGSM output is outside [0, 1].")
    if torch.max(torch.abs(x_pgd - inputs)).item() > 0.10001:
        raise RuntimeError("PGD perturbation exceeds epsilon.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_standard_one_step(model, inputs, targets, optimizer)
    train_fgsm_one_step(model, inputs, targets, optimizer, epsilon=0.1)

    smoke_dir = ensure_dir("results/smoke")
    robustness_csv = Path(smoke_dir) / "smoke_robustness.csv"
    write_csv(
        [
            {
                "model": "lenet_smoke",
                "epsilon": 0.0,
                "robust_accuracy": 0.5,
            },
            {
                "model": "lenet_smoke",
                "epsilon": 0.1,
                "robust_accuracy": 0.25,
            },
        ],
        robustness_csv,
    )
    plot_robustness_curve(robustness_csv, Path(smoke_dir) / "smoke_robustness.png")

    examples_path = Path(smoke_dir) / "smoke_examples.pt"
    torch.save(
        {
            "original": inputs[:4].cpu(),
            "fgsm": x_fgsm[:4].cpu(),
            "pgd": x_pgd[:4].cpu(),
            "labels": targets[:4].cpu(),
            "epsilon": 0.1,
        },
        examples_path,
    )
    plot_adversarial_examples(examples_path, Path(smoke_dir) / "smoke_examples.png")

    if not pd.read_csv(robustness_csv).shape[0] == 2:
        raise RuntimeError("Smoke CSV was not saved correctly.")
    LOGGER.info("Smoke test passed. Outputs written to %s", smoke_dir)


if __name__ == "__main__":
    main()
