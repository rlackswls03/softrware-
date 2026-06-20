"""CNN model definitions for MNIST experiments."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn


class LeNet(nn.Module):
    """A compact LeNet-style network for 28x28 grayscale MNIST images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, 10]``."""
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """A slightly deeper CNN that is structurally distinct from LeNet."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[batch, 10]``."""
        return self.classifier(self.features(x))


MODEL_ARCHITECTURES: Final[dict[str, type[nn.Module]]] = {
    "lenet": LeNet,
    "smallcnn": SmallCNN,
}

MODEL_SPECS: Final[dict[str, tuple[str, str]]] = {
    "lenet_standard": ("lenet", "standard"),
    "smallcnn_standard": ("smallcnn", "standard"),
    "lenet_fgsm_at": ("lenet", "fgsm_at"),
    "smallcnn_fgsm_at": ("smallcnn", "fgsm_at"),
}

MODEL_ORDER: Final[list[str]] = [
    "lenet_standard",
    "smallcnn_standard",
    "lenet_fgsm_at",
    "smallcnn_fgsm_at",
]


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    """Create a model by architecture name or experiment model key."""
    normalized = name.lower()
    if normalized in MODEL_SPECS:
        normalized = MODEL_SPECS[normalized][0]
    if normalized not in MODEL_ARCHITECTURES:
        valid = sorted([*MODEL_ARCHITECTURES, *MODEL_SPECS])
        raise ValueError(f"Unknown model name '{name}'. Valid names: {valid}")
    return MODEL_ARCHITECTURES[normalized](num_classes=num_classes)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def model_key_parts(model_key: str) -> tuple[str, str]:
    """Return ``(architecture, training_method)`` for a configured model key."""
    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown configured model key '{model_key}'.") from exc
