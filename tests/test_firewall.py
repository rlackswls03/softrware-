"""Tests for Adversarial Firewall detection and reject policy helpers."""

from __future__ import annotations

import torch

from adversarial_mnist.firewall import (
    ACCEPT_ORIGINAL,
    ACCEPT_PURIFIED,
    REJECT_SUSPICIOUS,
    binary_auc,
    choose_threshold_from_clean_validation,
    firewall_decision,
    firewall_decisions,
    reconstruction_error,
    tpr_at_fpr,
)


def test_reconstruction_error_shape_and_values() -> None:
    x = torch.zeros(3, 1, 2, 2)
    reconstructed = torch.ones_like(x) * 0.5
    scores = reconstruction_error(x, reconstructed)
    assert scores.shape == (3,)
    assert torch.allclose(scores, torch.full((3,), 0.25))


def test_threshold_selection_uses_clean_quantile() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    threshold = choose_threshold_from_clean_validation(scores, fpr=0.25)
    assert threshold == 2.25


def test_firewall_decision_policy() -> None:
    assert firewall_decision(0.1, threshold=0.2, purified_confidence=0.1) == ACCEPT_ORIGINAL
    assert firewall_decision(0.3, threshold=0.2, purified_confidence=0.8) == ACCEPT_PURIFIED
    assert firewall_decision(0.3, threshold=0.2, purified_confidence=0.6) == REJECT_SUSPICIOUS


def test_firewall_decisions_batch() -> None:
    decisions = firewall_decisions(
        torch.tensor([0.1, 0.3, 0.4]),
        threshold=0.2,
        purified_confidences=torch.tensor([0.1, 0.8, 0.2]),
    )
    assert decisions == [ACCEPT_ORIGINAL, ACCEPT_PURIFIED, REJECT_SUSPICIOUS]


def test_detection_auc_and_tpr_at_fpr() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert binary_auc(scores, labels) == 1.0
    assert tpr_at_fpr(scores, labels, max_fpr=0.0) == 1.0
