"""Tests for FGSM and PGD attacks."""

from __future__ import annotations

import pytest
import torch

from adversarial_mnist.attacks import fgsm_attack, pgd_linf_attack
from adversarial_mnist.models import create_model


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    x = torch.rand((4, 1, 28, 28), generator=generator)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    return x, y


def test_fgsm_epsilon_zero_returns_original() -> None:
    model = create_model("lenet")
    x, y = _batch()
    x_adv = fgsm_attack(model, x, y, epsilon=0.0)
    assert torch.allclose(x_adv, x)
    assert x_adv is not x


def test_fgsm_linf_bound_and_range() -> None:
    model = create_model("lenet")
    x, y = _batch()
    epsilon = 0.2
    x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
    assert torch.max(torch.abs(x_adv - x)).item() <= epsilon + 1e-6
    assert float(x_adv.min()) >= 0.0
    assert float(x_adv.max()) <= 1.0


def test_pgd_linf_bound_and_range() -> None:
    model = create_model("lenet")
    x, y = _batch()
    epsilon = 0.2
    x_adv = pgd_linf_attack(model, x, y, epsilon=epsilon, steps=4, random_start=True)
    assert torch.max(torch.abs(x_adv - x)).item() <= epsilon + 1e-6
    assert float(x_adv.min()) >= 0.0
    assert float(x_adv.max()) <= 1.0


def test_attacks_do_not_populate_input_or_parameter_gradients() -> None:
    model = create_model("lenet")
    x, y = _batch()
    x.requires_grad_(True)
    assert all(parameter.grad is None for parameter in model.parameters())
    _ = fgsm_attack(model, x, y, epsilon=0.1)
    assert x.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    _ = pgd_linf_attack(model, x, y, epsilon=0.1, steps=2)
    assert x.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_negative_epsilon_raises_error() -> None:
    model = create_model("lenet")
    x, y = _batch()
    with pytest.raises(ValueError, match="epsilon"):
        fgsm_attack(model, x, y, epsilon=-0.1)
    with pytest.raises(ValueError, match="epsilon"):
        pgd_linf_attack(model, x, y, epsilon=-0.1)
