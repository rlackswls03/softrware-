"""Detection and decision helpers for the Adversarial Firewall MVP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import torch
from torch import nn

ACCEPT_ORIGINAL: Final[str] = "ACCEPT_ORIGINAL"
ACCEPT_PURIFIED: Final[str] = "ACCEPT_PURIFIED"
REJECT_SUSPICIOUS: Final[str] = "REJECT_SUSPICIOUS"
DECISION_LABELS: Final[tuple[str, str, str]] = (
    ACCEPT_ORIGINAL,
    ACCEPT_PURIFIED,
    REJECT_SUSPICIOUS,
)


def reconstruction_error(inputs: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Return per-sample mean squared reconstruction error."""
    if inputs.shape != reconstructed.shape:
        raise ValueError("inputs and reconstructed tensors must have matching shapes.")
    if inputs.ndim < 2:
        raise ValueError("inputs must include a batch dimension.")
    return (inputs - reconstructed).pow(2).flatten(start_dim=1).mean(dim=1)


@torch.no_grad()
def autoencoder_reconstruction_error(
    autoencoder: nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(reconstruction, per_sample_error)`` for a batch."""
    original_training = autoencoder.training
    autoencoder.eval()
    reconstructed = autoencoder(inputs).clamp(0.0, 1.0)
    errors = reconstruction_error(inputs, reconstructed)
    autoencoder.train(original_training)
    return reconstructed.detach(), errors.detach()


def choose_threshold_from_clean_validation(
    scores: torch.Tensor | Sequence[float],
    fpr: float = 0.05,
) -> float:
    """Choose the clean-score quantile threshold for a target false-positive rate."""
    if not 0.0 <= fpr < 1.0:
        raise ValueError("fpr must be in the interval [0, 1).")
    score_tensor = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if score_tensor.numel() == 0:
        raise ValueError("At least one clean validation score is required.")
    quantile = 1.0 - fpr
    return float(torch.quantile(score_tensor, quantile).item())


def firewall_decision(
    score: float,
    threshold: float,
    purified_confidence: float,
    min_confidence: float = 0.7,
) -> str:
    """Return the reject-option decision for one sample."""
    if min_confidence < 0.0 or min_confidence > 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if score <= threshold:
        return ACCEPT_ORIGINAL
    if purified_confidence >= min_confidence:
        return ACCEPT_PURIFIED
    return REJECT_SUSPICIOUS


def firewall_decisions(
    scores: torch.Tensor,
    threshold: float,
    purified_confidences: torch.Tensor,
    min_confidence: float = 0.7,
) -> list[str]:
    """Return reject-option decisions for a batch."""
    if scores.shape != purified_confidences.shape:
        raise ValueError("scores and purified_confidences must have matching shapes.")
    return [
        firewall_decision(float(score), threshold, float(confidence), min_confidence)
        for score, confidence in zip(scores.flatten(), purified_confidences.flatten(), strict=True)
    ]


def binary_roc_curve(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FPR, TPR, and thresholds for attack labels where higher score means attack."""
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if score_array.shape != label_array.shape:
        raise ValueError("scores and labels must have matching shapes.")
    if score_array.size == 0:
        raise ValueError("At least one score is required.")
    positives = int((label_array == 1).sum())
    negatives = int((label_array == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires at least one positive and one negative sample.")

    thresholds = np.r_[np.inf, np.sort(np.unique(score_array))[::-1], -np.inf]
    tpr_values: list[float] = []
    fpr_values: list[float] = []
    for threshold in thresholds:
        predicted_positive = score_array >= threshold
        true_positive = int((predicted_positive & (label_array == 1)).sum())
        false_positive = int((predicted_positive & (label_array == 0)).sum())
        tpr_values.append(true_positive / positives)
        fpr_values.append(false_positive / negatives)
    return np.asarray(fpr_values), np.asarray(tpr_values), thresholds


def binary_auc(scores: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray) -> float:
    """Return ROC AUC for binary labels where attack is positive."""
    fpr, tpr, _ = binary_roc_curve(scores, labels)
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def tpr_at_fpr(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Return the largest TPR observed at or below ``max_fpr``."""
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1].")
    fpr, tpr, _ = binary_roc_curve(scores, labels)
    valid = tpr[fpr <= max_fpr]
    if valid.size == 0:
        return 0.0
    return float(valid.max())
