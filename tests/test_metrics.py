"""Tests for metric helpers."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from adversarial_mnist.metrics import (
    accuracy_from_predictions,
    aggregate_mean_std,
    clean_accuracy_retention,
    conditional_transfer_success_rate,
)


def test_accuracy_calculation() -> None:
    predictions = torch.tensor([1, 2, 3, 4])
    targets = torch.tensor([1, 0, 3, 0])
    assert accuracy_from_predictions(predictions, targets) == pytest.approx(0.5)


def test_clean_accuracy_retention() -> None:
    assert clean_accuracy_retention(0.81, 0.9) == pytest.approx(90.0)


def test_conditional_transfer_success_rate_hand_example() -> None:
    source_clean = torch.tensor([True, True, False, True])
    target_clean = torch.tensor([True, False, True, True])
    target_adv = torch.tensor([False, False, False, True])
    rate, denominator = conditional_transfer_success_rate(source_clean, target_clean, target_adv)
    assert denominator == 2
    assert rate == pytest.approx(0.5)


def test_conditional_transfer_success_rate_denominator_zero() -> None:
    source_clean = torch.tensor([False, False])
    target_clean = torch.tensor([True, False])
    target_adv = torch.tensor([False, False])
    with pytest.warns(UserWarning, match="denominator is zero"):
        rate, denominator = conditional_transfer_success_rate(source_clean, target_clean, target_adv)
    assert denominator == 0
    assert math.isnan(rate)


def test_aggregate_mean_std() -> None:
    frame = pd.DataFrame(
        [
            {"model": "a", "epsilon": 0.1, "seed": 1, "robust_accuracy": 0.5},
            {"model": "a", "epsilon": 0.1, "seed": 2, "robust_accuracy": 0.7},
        ]
    )
    result = aggregate_mean_std(
        frame,
        group_columns=["model", "epsilon"],
        metric_columns=["robust_accuracy"],
    )
    assert result.loc[0, "robust_accuracy_mean"] == pytest.approx(0.6)
    assert result.loc[0, "robust_accuracy_std"] == pytest.approx(0.141421356, rel=1e-5)
