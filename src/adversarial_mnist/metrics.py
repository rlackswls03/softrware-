"""Metric helpers for adversarial robustness experiments."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy from logits and labels."""
    predictions = logits.argmax(dim=1)
    return accuracy_from_predictions(predictions, targets)


def accuracy_from_predictions(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy from predicted labels and labels."""
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    if targets.numel() == 0:
        return math.nan
    return (predictions == targets).float().mean().item()


def robust_accuracy(correct_adv: int, total: int) -> float:
    """Return robust accuracy from adversarially correct count and total count."""
    if total <= 0:
        return math.nan
    return correct_adv / total


def untargeted_attack_success_rate(adv_correct: int, total: int) -> float:
    """Return untargeted attack success rate over all evaluated samples."""
    if total <= 0:
        return math.nan
    return 1.0 - robust_accuracy(adv_correct, total)


def clean_accuracy_retention(defended_clean: float, standard_clean: float) -> float:
    """Return defended clean accuracy as a percent of its standard counterpart."""
    if standard_clean == 0:
        warnings.warn("Standard clean accuracy is zero; retention is undefined.", stacklevel=2)
        return math.nan
    return defended_clean / standard_clean * 100.0


def conditional_transfer_success_rate(
    source_clean_correct: torch.Tensor,
    target_clean_correct: torch.Tensor,
    target_adv_correct: torch.Tensor,
) -> tuple[float, int]:
    """Return conditional transfer success rate and denominator.

    The denominator is samples correctly classified by both source and target on
    clean inputs. The numerator is those samples misclassified by target after
    applying source-generated adversarial examples.
    """
    joint_clean = source_clean_correct.bool() & target_clean_correct.bool()
    denominator = int(joint_clean.sum().item())
    if denominator == 0:
        message = "Conditional transfer denominator is zero; returning NaN."
        warnings.warn(message, stacklevel=2)
        LOGGER.warning(message)
        return math.nan, 0
    target_failure = joint_clean & ~target_adv_correct.bool()
    numerator = int(target_failure.sum().item())
    return numerator / denominator, denominator


def aggregate_mean_std(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate metric columns over seeds with mean and sample std."""
    if frame.empty:
        return pd.DataFrame()
    missing = set(group_columns).union(metric_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot aggregate; missing columns: {sorted(missing)}")
    grouped = frame.groupby(list(group_columns), dropna=False)[list(metric_columns)]
    aggregated = grouped.agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns.to_flat_index()
    ]
    return aggregated


def mean_std_dict(values: Sequence[float]) -> dict[str, float]:
    """Return mean and sample standard deviation for a numeric sequence."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": math.nan, "std": math.nan}
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array, ddof=1)) if array.size > 1 else 0.0,
    }


def records_to_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from records with a clear empty-record behavior."""
    return pd.DataFrame(list(records))
