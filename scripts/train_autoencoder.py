"""Train MNIST convolutional autoencoders for the Adversarial Firewall."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from adversarial_mnist.autoencoder import (
    ConvAutoencoder,
    autoencoder_checkpoint_path,
    save_autoencoder_checkpoint,
)
from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import write_csv
from adversarial_mnist.utils import set_reproducibility, setup_logging
from scripts.common import add_common_args, prepare_config, prepare_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=None, help="Override autoencoder epochs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


@torch.no_grad()
def evaluate_autoencoder_loss(
    autoencoder: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Return average validation MSE reconstruction loss."""
    original_training = autoencoder.training
    autoencoder.eval()
    total_loss = 0.0
    total = 0
    for inputs, _ in loader:
        inputs = inputs.to(device)
        reconstructed = autoencoder(inputs)
        loss = F.mse_loss(reconstructed, inputs)
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.item()) * batch_size
        total += batch_size
    autoencoder.train(original_training)
    if total == 0:
        raise RuntimeError("Autoencoder validation loader produced zero samples.")
    return total_loss / total


def weighted_mse_objective(reconstructed: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Return a stroke-aware MSE objective for MNIST autoencoder training."""
    weights = 1.0 + 4.0 * inputs
    return ((reconstructed - inputs).pow(2) * weights).mean()


def train_autoencoder_for_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    force: bool = False,
    progress: bool = True,
) -> tuple[ConvAutoencoder, list[dict[str, Any]]]:
    """Train or load one seed-specific autoencoder."""
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    checkpoint = autoencoder_checkpoint_path(checkpoints_dir, seed)
    autoencoder = ConvAutoencoder().to(device)
    if checkpoint.exists() and not force:
        LOGGER.info("Skipping autoencoder seed=%s because %s exists.", seed, checkpoint)
        payload = torch.load(checkpoint, map_location=device)
        autoencoder.load_state_dict(payload["model_state_dict"])
        autoencoder.eval()
        return autoencoder, []

    loaders = create_mnist_dataloaders(config, seed=seed)
    ae_config = config.get("firewall", {}).get("autoencoder", {})
    epochs = int(ae_config.get("epochs", 5))
    learning_rate = float(ae_config.get("learning_rate", 0.001))
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        autoencoder.train()
        total_loss = 0.0
        total = 0
        iterator = tqdm(
            loaders.train,
            desc=f"autoencoder seed {seed} epoch {epoch}",
            leave=False,
            disable=not progress,
        )
        for inputs, _ in iterator:
            inputs = inputs.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed = autoencoder(inputs)
            reconstruction_mse = F.mse_loss(reconstructed, inputs)
            loss = weighted_mse_objective(reconstructed, inputs)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered during autoencoder training.")
            loss.backward()
            optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(reconstruction_mse.item()) * batch_size
            total += batch_size

        if total == 0:
            raise RuntimeError("Autoencoder training loader produced zero samples.")
        train_loss = total_loss / total
        validation_loss = evaluate_autoencoder_loss(autoencoder, loaders.validation, device)
        row = {
            "seed": seed,
            "epoch": epoch,
            "train_reconstruction_loss": train_loss,
            "validation_reconstruction_loss": validation_loss,
            "runtime_seconds": perf_counter() - start,
            "device": str(device),
        }
        rows.append(row)
        LOGGER.info(
            "autoencoder seed=%s epoch=%s validation_reconstruction_loss=%.6f",
            seed,
            epoch,
            validation_loss,
        )

    save_autoencoder_checkpoint(
        checkpoint,
        autoencoder,
        optimizer,
        {
            "model_key": "autoencoder",
            "seed": seed,
            "epoch": epochs,
            "config": config,
            "validation_reconstruction_loss": rows[-1]["validation_reconstruction_loss"],
        },
    )
    autoencoder.eval()
    return autoencoder, rows


def train_autoencoders(
    config: dict[str, Any],
    device: torch.device,
    force: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Train autoencoders for all configured seeds and return history rows."""
    rows: list[dict[str, Any]] = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        set_reproducibility(seed, deterministic=bool(config["training"]["deterministic"]))
        _, seed_rows = train_autoencoder_for_seed(
            config,
            seed,
            device,
            force=force,
            progress=progress,
        )
        rows.extend(seed_rows)
    return rows


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    if args.epochs is not None:
        config.setdefault("firewall", {}).setdefault("autoencoder", {})["epochs"] = args.epochs
    device = prepare_device(args)
    output_csv = Path(config["paths"]["raw_dir"]) / "autoencoder_history.csv"
    if args.force and output_csv.exists():
        output_csv.unlink()
    rows = train_autoencoders(config, device, force=args.force, progress=not args.no_progress)
    if rows:
        write_csv(rows, output_csv)
        LOGGER.info("Wrote %s autoencoder history rows to %s.", len(rows), output_csv)
    else:
        LOGGER.info("No new autoencoder training rows were produced.")


if __name__ == "__main__":
    main()
