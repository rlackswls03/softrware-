"""FGSM and PGD L-infinity attacks."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Attack model must have at least one parameter.") from exc


def _validate_attack_inputs(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> None:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if not x.is_floating_point():
        raise ValueError("Input tensor x must be floating point.")
    model_device = _model_device(model)
    if x.device != model_device:
        raise ValueError(f"Input device {x.device} does not match model device {model_device}.")
    if y.device != model_device:
        raise ValueError(f"Target device {y.device} does not match model device {model_device}.")
    if x.ndim < 2:
        raise ValueError("Input tensor x must include a batch dimension.")
    if y.ndim != 1:
        raise ValueError("Target tensor y must have shape [batch].")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input and target batch sizes differ.")


@contextlib.contextmanager
def _temporary_eval(model: nn.Module, enabled: bool = True):
    original_training = model.training
    if enabled:
        model.eval()
    try:
        yield
    finally:
        if enabled:
            model.train(original_training)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: LossFn = F.cross_entropy,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    use_eval_mode: bool = True,
) -> torch.Tensor:
    """Create untargeted FGSM examples in `[clamp_min, clamp_max]`.

    The model parameters are not updated and parameter ``.grad`` fields are not
    populated by this function.
    """
    _validate_attack_inputs(model, x, y, epsilon)
    if epsilon == 0:
        return x.detach().clone()

    with _temporary_eval(model, enabled=use_eval_mode):
        x_for_grad = x.detach().clone().requires_grad_(True)
        logits = model(x_for_grad)
        loss = loss_fn(logits, y)
        (gradient_x,) = torch.autograd.grad(loss, x_for_grad, only_inputs=True)
        x_adv = x_for_grad + epsilon * gradient_x.sign()
        return torch.clamp(x_adv, clamp_min, clamp_max).detach()


def pgd_linf_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.25,
    steps: int = 10,
    alpha: float | None = None,
    random_start: bool = True,
    loss_fn: LossFn = F.cross_entropy,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    use_eval_mode: bool = True,
) -> torch.Tensor:
    """Create untargeted PGD L-infinity adversarial examples."""
    _validate_attack_inputs(model, x, y, epsilon)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if epsilon == 0:
        return x.detach().clone()
    step_size = epsilon / steps if alpha is None else alpha
    if step_size <= 0:
        raise ValueError("alpha must be positive.")

    x_original = x.detach()
    if random_start:
        perturbation = torch.empty_like(x_original).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_original + perturbation, clamp_min, clamp_max).detach()
    else:
        x_adv = x_original.clone().detach()

    with _temporary_eval(model, enabled=use_eval_mode):
        for _ in range(steps):
            x_for_grad = x_adv.detach().clone().requires_grad_(True)
            logits = model(x_for_grad)
            loss = loss_fn(logits, y)
            (gradient_x,) = torch.autograd.grad(loss, x_for_grad, only_inputs=True)
            x_adv = x_for_grad + step_size * gradient_x.sign()
            eta = torch.clamp(x_adv - x_original, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_original + eta, clamp_min, clamp_max).detach()
    return x_adv.detach()


def pgd_linf_attack_restarts(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.25,
    steps: int = 20,
    restarts: int = 5,
    alpha: float | None = None,
    loss_fn: LossFn = F.cross_entropy,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    use_eval_mode: bool = True,
) -> torch.Tensor:
    """Run PGD multiple times and keep the highest-loss example per sample."""
    _validate_attack_inputs(model, x, y, epsilon)
    if restarts <= 0:
        raise ValueError("restarts must be positive.")
    if epsilon == 0:
        return x.detach().clone()

    best_adv: torch.Tensor | None = None
    best_losses: torch.Tensor | None = None
    for _ in range(restarts):
        x_adv = pgd_linf_attack(
            model,
            x,
            y,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            random_start=True,
            loss_fn=loss_fn,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            use_eval_mode=use_eval_mode,
        )
        with _temporary_eval(model, enabled=use_eval_mode), torch.no_grad():
            logits = model(x_adv)
            losses = F.cross_entropy(logits, y, reduction="none")
        if best_adv is None or best_losses is None:
            best_adv = x_adv
            best_losses = losses
            continue
        replace = losses > best_losses
        best_losses = torch.where(replace, losses, best_losses)
        best_adv = torch.where(replace.view(-1, *([1] * (x_adv.ndim - 1))), x_adv, best_adv)

    if best_adv is None:
        raise RuntimeError("PGD restart attack did not produce adversarial examples.")
    return best_adv.detach()
