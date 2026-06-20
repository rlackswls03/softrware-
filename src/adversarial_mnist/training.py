"""Training loops and checkpoint helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from adversarial_mnist.attacks import fgsm_attack
from adversarial_mnist.models import create_model, model_key_parts
from adversarial_mnist.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def checkpoint_path(
    checkpoints_dir: str | Path,
    model_key: str,
    seed: int,
    kind: str = "last",
) -> Path:
    """Return the checkpoint path for a model/seed/kind tuple."""
    return Path(checkpoints_dir) / f"{model_key}_seed{seed}_{kind}.pt"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metadata: dict[str, Any],
) -> None:
    """Save a model checkpoint with metadata."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint weights and optional optimizer state."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata in {path} is not a dictionary.")
    return metadata


def evaluate_loss_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Evaluate average loss and accuracy without modifying model state."""
    criterion = criterion or nn.CrossEntropyLoss()
    original_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered during evaluation.")
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size
    model.train(original_training)
    if total == 0:
        raise RuntimeError("Evaluation loader produced zero samples.")
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
    }


def _train_standard_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    progress: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    iterator = tqdm(loader, desc=f"standard epoch {epoch}", leave=False, disable=not progress)
    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during standard training.")
        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return {
        "train_loss": total_loss / total,
        "train_accuracy": total_correct / total,
        "train_clean_loss": total_loss / total,
        "train_adversarial_loss": float("nan"),
    }


def _train_fgsm_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epsilon: float,
    clean_weight: float,
    adversarial_weight: float,
    progress: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_clean_loss = 0.0
    total_adv_loss = 0.0
    total_correct = 0
    total = 0
    iterator = tqdm(loader, desc=f"fgsm_at epoch {epoch}", leave=False, disable=not progress)
    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        x_adv = fgsm_attack(model, inputs, targets, epsilon=epsilon, use_eval_mode=True)
        optimizer.zero_grad(set_to_none=True)

        logits_clean = model(inputs)
        logits_adv = model(x_adv)
        clean_loss = F.cross_entropy(logits_clean, targets)
        adv_loss = F.cross_entropy(logits_adv, targets)
        loss = clean_weight * clean_loss + adversarial_weight * adv_loss
        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during FGSM adversarial training.")
        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_clean_loss += float(clean_loss.item()) * batch_size
        total_adv_loss += float(adv_loss.item()) * batch_size
        total_correct += int((logits_clean.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return {
        "train_loss": total_loss / total,
        "train_accuracy": total_correct / total,
        "train_clean_loss": total_clean_loss / total,
        "train_adversarial_loss": total_adv_loss / total,
    }


def train_model(
    model_key: str,
    config: dict[str, Any],
    seed: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    force: bool = False,
    resume: bool = False,
    progress: bool = True,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train a configured model and return the model plus epoch history rows."""
    architecture, training_method = model_key_parts(model_key)
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    last_path = checkpoint_path(checkpoints_dir, model_key, seed, "last")
    best_path = checkpoint_path(checkpoints_dir, model_key, seed, "best_val")
    if last_path.exists() and not force and not resume:
        LOGGER.info("Skipping %s seed %s because %s exists.", model_key, seed, last_path)
        model = create_model(architecture).to(device)
        load_checkpoint(last_path, model, map_location=device)
        return model, []

    model = create_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    start_epoch = 1
    best_validation_accuracy = -1.0
    if resume and last_path.exists():
        metadata = load_checkpoint(last_path, model, optimizer=optimizer, map_location=device)
        start_epoch = int(metadata.get("epoch", 0)) + 1
        best_validation_accuracy = float(metadata.get("best_validation_accuracy", -1.0))

    epochs = int(config["training"]["epochs"])
    fgsm_epsilon = float(config["training"]["fgsm_adversarial_epsilon"])
    clean_weight = float(config["training"]["clean_loss_weight"])
    adversarial_weight = float(config["training"]["adversarial_loss_weight"])

    rows: list[dict[str, Any]] = []
    for epoch in range(start_epoch, epochs + 1):
        start = perf_counter()
        if training_method == "standard":
            train_metrics = _train_standard_epoch(
                model, train_loader, optimizer, device, epoch, progress=progress
            )
        elif training_method == "fgsm_at":
            train_metrics = _train_fgsm_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                epsilon=fgsm_epsilon,
                clean_weight=clean_weight,
                adversarial_weight=adversarial_weight,
                progress=progress,
            )
        else:
            raise ValueError(f"Unknown training method '{training_method}'.")

        validation_metrics = evaluate_loss_accuracy(model, validation_loader, device)
        runtime_seconds = perf_counter() - start
        row = {
            "model": model_key,
            "architecture": architecture,
            "training": training_method,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_clean_loss": train_metrics["train_clean_loss"],
            "train_adversarial_loss": train_metrics["train_adversarial_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "fgsm_train_epsilon": fgsm_epsilon if training_method == "fgsm_at" else float("nan"),
            "runtime_seconds": runtime_seconds,
            "device": str(device),
        }
        rows.append(row)

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            save_checkpoint(
                best_path,
                model,
                optimizer,
                {
                    "model_key": model_key,
                    "model_name": architecture,
                    "training": training_method,
                    "seed": seed,
                    "epoch": epoch,
                    "epsilon": fgsm_epsilon if training_method == "fgsm_at" else None,
                    "best_validation_accuracy": best_validation_accuracy,
                    "config": config,
                },
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            {
                "model_key": model_key,
                "model_name": architecture,
                "training": training_method,
                "seed": seed,
                "epoch": epoch,
                "epsilon": fgsm_epsilon if training_method == "fgsm_at" else None,
                "best_validation_accuracy": best_validation_accuracy,
                "config": config,
            },
        )
        LOGGER.info(
            "%s seed=%s epoch=%s validation_accuracy=%.4f",
            model_key,
            seed,
            epoch,
            validation_metrics["accuracy"],
        )
    return model, rows


def train_standard_one_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one standard training step for smoke tests."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_fgsm_one_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.25,
) -> float:
    """Run one FGSM adversarial training step for smoke tests."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x_adv = fgsm_attack(model, inputs, targets, epsilon=epsilon, use_eval_mode=True)
    optimizer.zero_grad(set_to_none=True)
    logits_clean = model(inputs)
    logits_adv = model(x_adv)
    loss = 0.5 * F.cross_entropy(logits_clean, targets) + 0.5 * F.cross_entropy(
        logits_adv, targets
    )
    loss.backward()
    optimizer.step()
    return float(loss.item())
