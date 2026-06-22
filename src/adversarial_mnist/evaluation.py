"""Evaluation routines for clean, FGSM, transfer, and PGD experiments."""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from adversarial_mnist.attacks import fgsm_attack, pgd_linf_attack, pgd_linf_attack_restarts
from adversarial_mnist.metrics import (
    aggregate_mean_std,
    robust_accuracy,
    untargeted_attack_success_rate,
)
from adversarial_mnist.models import MODEL_ORDER, create_model, model_key_parts
from adversarial_mnist.training import checkpoint_path, load_checkpoint
from adversarial_mnist.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


def _training_from_key(model_key: str) -> str:
    return model_key_parts(model_key)[1]


def _architecture_from_key(model_key: str) -> str:
    return model_key_parts(model_key)[0]


def _predict_correct(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(inputs)
        return logits.argmax(dim=1) == targets


def load_trained_models(
    model_keys: list[str],
    checkpoints_dir: str | Path,
    seed: int,
    device: torch.device,
    checkpoint_kind: str = "last",
) -> dict[str, nn.Module]:
    """Load trained models from checkpoints into eval mode."""
    models: dict[str, nn.Module] = {}
    for model_key in model_keys:
        architecture = _architecture_from_key(model_key)
        model = create_model(architecture).to(device)
        path = checkpoint_path(checkpoints_dir, model_key, seed, checkpoint_kind)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_key} seed {seed}: {path}")
        load_checkpoint(path, model, map_location=device)
        model.eval()
        models[model_key] = model
    return models


def evaluate_clean_accuracy(
    model_key: str,
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate clean accuracy for one model."""
    start = perf_counter()
    model.eval()
    total = 0
    correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        correct_tensor = _predict_correct(model, inputs, targets)
        correct += int(correct_tensor.sum().item())
        total += int(targets.shape[0])
    if total == 0:
        raise RuntimeError("Clean evaluation loader produced zero samples.")
    return {
        "model": model_key,
        "architecture": _architecture_from_key(model_key),
        "training": _training_from_key(model_key),
        "seed": seed,
        "evaluated_samples": total,
        "clean_accuracy": correct / total,
        "runtime_seconds": perf_counter() - start,
        "device": str(device),
    }


def evaluate_all_clean(
    models: dict[str, nn.Module],
    loader: DataLoader,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate clean accuracy for all configured models."""
    return [
        evaluate_clean_accuracy(model_key, model, loader, seed, device)
        for model_key, model in models.items()
    ]


def evaluate_fgsm_robustness(
    model_key: str,
    model: nn.Module,
    loader: DataLoader,
    epsilons: list[float],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate white-box FGSM robust accuracy over epsilon values."""
    rows: list[dict[str, Any]] = []
    model.eval()
    for epsilon in epsilons:
        start = perf_counter()
        total = 0
        adv_correct = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            x_adv = fgsm_attack(model, inputs, targets, epsilon=float(epsilon))
            target_adv_correct = _predict_correct(model, x_adv, targets)
            adv_correct += int(target_adv_correct.sum().item())
            total += int(targets.shape[0])
        robust_acc = robust_accuracy(adv_correct, total)
        rows.append(
            {
                "model": model_key,
                "architecture": _architecture_from_key(model_key),
                "training": _training_from_key(model_key),
                "attack": "FGSM",
                "epsilon": float(epsilon),
                "seed": seed,
                "evaluated_samples": total,
                "robust_accuracy": robust_acc,
                "attack_success_rate": untargeted_attack_success_rate(adv_correct, total),
                "runtime_seconds": perf_counter() - start,
                "device": str(device),
            }
        )
    return rows


def evaluate_all_fgsm_robustness(
    models: dict[str, nn.Module],
    loader: DataLoader,
    epsilons: list[float],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Evaluate FGSM robustness for all models."""
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        rows.extend(evaluate_fgsm_robustness(model_key, model, loader, epsilons, seed, device))
    return rows


def evaluate_transferability(
    models: dict[str, nn.Module],
    loader: DataLoader,
    epsilons: list[float],
    seed: int,
    device: torch.device,
    model_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate source-target FGSM transferability matrices in long form."""
    order = model_order or MODEL_ORDER
    missing = [model_key for model_key in order if model_key not in models]
    if missing:
        raise ValueError(f"Missing models for transfer evaluation: {missing}")
    for model in models.values():
        model.eval()

    rows: list[dict[str, Any]] = []
    for epsilon in epsilons:
        stats: dict[tuple[str, str], dict[str, float]] = {
            (source_key, target_key): {
                "evaluated_samples": 0.0,
                "jointly_clean_correct_samples": 0.0,
                "target_adv_correct": 0.0,
                "conditional_transfer_success_count": 0.0,
                "runtime_seconds": 0.0,
            }
            for source_key in order
            for target_key in order
        }
        for source_key in order:
            source_model = models[source_key]
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                source_clean_correct = _predict_correct(source_model, inputs, targets)
                adv_start = perf_counter()
                x_adv = fgsm_attack(source_model, inputs, targets, epsilon=float(epsilon))
                adv_runtime = perf_counter() - adv_start
                for target_key in order:
                    target_model = models[target_key]
                    target_start = perf_counter()
                    target_clean_correct = _predict_correct(target_model, inputs, targets)
                    target_adv_correct = _predict_correct(target_model, x_adv, targets)
                    target_runtime = perf_counter() - target_start

                    joint_clean = source_clean_correct & target_clean_correct
                    pair_stats = stats[(source_key, target_key)]
                    pair_stats["evaluated_samples"] += int(targets.shape[0])
                    pair_stats["jointly_clean_correct_samples"] += int(joint_clean.sum().item())
                    pair_stats["target_adv_correct"] += int(target_adv_correct.sum().item())
                    pair_stats["conditional_transfer_success_count"] += int(
                        (joint_clean & ~target_adv_correct).sum().item()
                    )
                    pair_stats["runtime_seconds"] += adv_runtime + target_runtime

        for source_key in order:
            for target_key in order:
                pair_stats = stats[(source_key, target_key)]
                evaluated = int(pair_stats["evaluated_samples"])
                adv_correct = int(pair_stats["target_adv_correct"])
                jointly_clean = int(pair_stats["jointly_clean_correct_samples"])
                if jointly_clean == 0:
                    conditional_success = math.nan
                    warning = (
                        "Conditional transfer denominator is zero for "
                        f"{source_key}->{target_key} epsilon={epsilon} seed={seed}."
                    )
                    warnings.warn(warning, stacklevel=2)
                    LOGGER.warning(warning)
                else:
                    conditional_success = (
                        pair_stats["conditional_transfer_success_count"] / jointly_clean
                    )
                rows.append(
                    {
                        "source_model": source_key,
                        "target_model": target_key,
                        "source_training": _training_from_key(source_key),
                        "target_training": _training_from_key(target_key),
                        "attack": "FGSM",
                        "epsilon": float(epsilon),
                        "seed": seed,
                        "evaluated_samples": evaluated,
                        "jointly_clean_correct_samples": jointly_clean,
                        "robust_accuracy": robust_accuracy(adv_correct, evaluated),
                        "attack_success_rate": untargeted_attack_success_rate(
                            adv_correct, evaluated
                        ),
                        "conditional_transfer_success_rate": conditional_success,
                        "runtime_seconds": pair_stats["runtime_seconds"],
                        "device": str(device),
                    }
                )
    return rows


def evaluate_pgd_whitebox(
    model_key: str,
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    device: torch.device,
    epsilon: float = 0.25,
    steps: int = 10,
    alpha: float | None = None,
    random_start: bool = True,
) -> dict[str, Any]:
    """Evaluate white-box PGD robust accuracy for one model."""
    start = perf_counter()
    model.eval()
    total = 0
    adv_correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        x_adv = pgd_linf_attack(
            model,
            inputs,
            targets,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            random_start=random_start,
        )
        target_adv_correct = _predict_correct(model, x_adv, targets)
        adv_correct += int(target_adv_correct.sum().item())
        total += int(targets.shape[0])
    robust_acc = robust_accuracy(adv_correct, total)
    return {
        "model": model_key,
        "architecture": _architecture_from_key(model_key),
        "training": _training_from_key(model_key),
        "attack": "PGD_LINF",
        "epsilon": float(epsilon),
        "pgd_steps": int(steps),
        "pgd_alpha": float(epsilon / steps if alpha is None else alpha),
        "pgd_random_start": bool(random_start),
        "seed": seed,
        "evaluated_samples": total,
        "robust_accuracy": robust_acc,
        "attack_success_rate": untargeted_attack_success_rate(adv_correct, total),
        "runtime_seconds": perf_counter() - start,
        "device": str(device),
    }


def evaluate_all_pgd_whitebox(
    models: dict[str, nn.Module],
    loader: DataLoader,
    seed: int,
    device: torch.device,
    pgd_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate PGD white-box robustness for all models."""
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        rows.append(
            evaluate_pgd_whitebox(
                model_key,
                model,
                loader,
                seed,
                device,
                epsilon=float(pgd_config.get("epsilon", 0.25)),
                steps=int(pgd_config.get("steps", 10)),
                alpha=pgd_config.get("alpha"),
                random_start=bool(pgd_config.get("random_start", True)),
            )
        )
    return rows


def evaluate_pgd_whitebox_restarts(
    model_key: str,
    model: nn.Module,
    loader: DataLoader,
    seed: int,
    device: torch.device,
    epsilon: float = 0.25,
    steps: int = 20,
    restarts: int = 5,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Evaluate PGD white-box robustness with multiple random restarts."""
    start = perf_counter()
    model.eval()
    total = 0
    adv_correct = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        x_adv = pgd_linf_attack_restarts(
            model,
            inputs,
            targets,
            epsilon=epsilon,
            steps=steps,
            restarts=restarts,
            alpha=alpha,
        )
        target_adv_correct = _predict_correct(model, x_adv, targets)
        adv_correct += int(target_adv_correct.sum().item())
        total += int(targets.shape[0])
    robust_acc = robust_accuracy(adv_correct, total)
    return {
        "model": model_key,
        "architecture": _architecture_from_key(model_key),
        "training": _training_from_key(model_key),
        "attack": "PGD_LINF_RESTARTS",
        "epsilon": float(epsilon),
        "pgd_steps": int(steps),
        "pgd_restarts": int(restarts),
        "pgd_alpha": float(epsilon / steps if alpha is None else alpha),
        "pgd_random_start": True,
        "seed": seed,
        "evaluated_samples": total,
        "robust_accuracy": robust_acc,
        "attack_success_rate": untargeted_attack_success_rate(adv_correct, total),
        "runtime_seconds": perf_counter() - start,
        "device": str(device),
    }


def evaluate_all_pgd_whitebox_restarts(
    models: dict[str, nn.Module],
    loader: DataLoader,
    seed: int,
    device: torch.device,
    epsilon: float = 0.25,
    steps: int = 20,
    restarts: int = 5,
    alpha: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate multi-restart PGD white-box robustness for all models."""
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        rows.append(
            evaluate_pgd_whitebox_restarts(
                model_key,
                model,
                loader,
                seed,
                device,
                epsilon=epsilon,
                steps=steps,
                restarts=restarts,
                alpha=alpha,
            )
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write evaluation rows to CSV."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def aggregate_result_files(raw_dir: str | Path, aggregated_dir: str | Path) -> None:
    """Create multi-seed mean/std aggregate CSV files when raw files exist."""
    raw_path = Path(raw_dir)
    output_path = ensure_dir(aggregated_dir)

    clean_file = raw_path / "clean_accuracy.csv"
    if clean_file.exists():
        clean = pd.read_csv(clean_file)
        aggregate_mean_std(
            clean,
            group_columns=["model", "architecture", "training"],
            metric_columns=["clean_accuracy"],
        ).to_csv(output_path / "model_summary.csv", index=False)

    robustness_file = raw_path / "fgsm_robustness.csv"
    if robustness_file.exists():
        robustness = pd.read_csv(robustness_file)
        aggregate_mean_std(
            robustness,
            group_columns=["model", "architecture", "training", "attack", "epsilon"],
            metric_columns=["robust_accuracy", "attack_success_rate"],
        ).to_csv(output_path / "robustness_summary.csv", index=False)

    transfer_file = raw_path / "transferability_long.csv"
    if transfer_file.exists():
        transfer = pd.read_csv(transfer_file)
        aggregate_mean_std(
            transfer,
            group_columns=[
                "source_model",
                "target_model",
                "source_training",
                "target_training",
                "attack",
                "epsilon",
            ],
            metric_columns=[
                "robust_accuracy",
                "attack_success_rate",
                "conditional_transfer_success_rate",
            ],
        ).to_csv(output_path / "transferability_summary.csv", index=False)


def save_adversarial_example_batch(
    model: nn.Module,
    loader: DataLoader,
    path: str | Path,
    device: torch.device,
    epsilon: float = 0.25,
    pgd_steps: int = 10,
    max_examples: int = 8,
) -> None:
    """Save one batch of original, FGSM, and PGD examples for plotting."""
    model.eval()
    output_path = Path(path)
    ensure_dir(output_path.parent)
    inputs, targets = next(iter(loader))
    inputs = inputs[:max_examples].to(device)
    targets = targets[:max_examples].to(device)
    x_fgsm = fgsm_attack(model, inputs, targets, epsilon=epsilon)
    x_pgd = pgd_linf_attack(model, inputs, targets, epsilon=epsilon, steps=pgd_steps)
    with torch.no_grad():
        clean_predictions = model(inputs).argmax(dim=1)
        fgsm_predictions = model(x_fgsm).argmax(dim=1)
        pgd_predictions = model(x_pgd).argmax(dim=1)
    torch.save(
        {
            "original": inputs.cpu(),
            "fgsm": x_fgsm.cpu(),
            "pgd": x_pgd.cpu(),
            "labels": targets.cpu(),
            "clean_predictions": clean_predictions.cpu(),
            "fgsm_predictions": fgsm_predictions.cpu(),
            "pgd_predictions": pgd_predictions.cpu(),
            "epsilon": epsilon,
        },
        output_path,
    )
