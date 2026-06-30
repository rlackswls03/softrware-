from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from adversarial_mnist.firewall import ACCEPT_ORIGINAL, ACCEPT_PURIFIED
from adversarial_mnist.simulation import build_simulation


def _condition_batch(count: int = 2) -> dict[str, object]:
    base = torch.zeros(count, 1, 28, 28)
    base[:, :, 8:20, 10:18] = 0.8
    attacked = (base + 0.1).clamp(0.0, 1.0)
    purified = (attacked - 0.05).clamp(0.0, 1.0)
    return {
        "original": base,
        "input": attacked,
        "purified": purified,
        "labels": torch.tensor([1, 2]),
        "clean_predictions": torch.tensor([1, 2]),
        "input_predictions": torch.tensor([8, 2]),
        "purified_predictions": torch.tensor([1, 2]),
        "scores": torch.tensor([0.01, 0.02]),
        "decisions": [ACCEPT_PURIFIED, ACCEPT_ORIGINAL],
    }


def test_build_simulation_from_saved_artifacts(tmp_path: Path) -> None:
    examples_path = tmp_path / "firewall_examples.pt"
    metrics_path = tmp_path / "firewall_results.csv"
    torch.save(
        {
            "model": "smallcnn_standard",
            "seed": 42,
            "epsilon": 0.25,
            "threshold": 0.015,
            "min_confidence": 0.7,
            "conditions": {
                "Clean": _condition_batch(),
                "FGSM": _condition_batch(),
                "PGD": _condition_batch(),
            },
        },
        examples_path,
    )
    pd.DataFrame(
        [
            {
                "model": "smallcnn_standard",
                "seed": 42,
                "condition": condition,
                "original_accuracy": 0.5,
                "purified_accuracy": 0.75,
                "detection_rate": 1.0,
                "reject_rate": 0.1,
                "final_safe_accuracy": 0.8,
                "evaluated_samples": 2,
            }
            for condition in ("Clean", "FGSM", "PGD")
        ]
    ).to_csv(metrics_path, index=False)
    config = {
        "paths": {
            "raw_dir": str(tmp_path),
            "results_dir": str(tmp_path / "results"),
        }
    }

    output = build_simulation(config, examples_path=examples_path, metrics_path=metrics_path)

    html = output.read_text(encoding="utf-8")
    assert output == tmp_path / "results" / "simulation" / "index.html"
    assert "const SIM_DATA =" in html
    assert "Adversarial Firewall Simulation" in html
    assert html.count("data:image/png;base64,") == 18
