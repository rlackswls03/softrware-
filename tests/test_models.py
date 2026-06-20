"""Tests for model construction."""

from __future__ import annotations

import pytest
import torch

from adversarial_mnist.models import count_parameters, create_model


def test_model_output_shapes() -> None:
    x = torch.rand(4, 1, 28, 28)
    for name in ("lenet", "smallcnn"):
        model = create_model(name)
        logits = model(x)
        assert logits.shape == (4, 10)


def test_models_have_different_parameter_counts() -> None:
    lenet = create_model("lenet")
    smallcnn = create_model("smallcnn")
    assert count_parameters(lenet) != count_parameters(smallcnn)


def test_invalid_model_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        create_model("not_a_model")
