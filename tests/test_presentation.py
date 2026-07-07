from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from adversarial_mnist.firewall import ACCEPT_ORIGINAL, ACCEPT_PURIFIED
from adversarial_mnist.presentation import build_presentation

MODELS = [
    ("lenet_standard", "lenet", "standard"),
    ("smallcnn_standard", "smallcnn", "standard"),
    ("lenet_fgsm_at", "lenet", "fgsm_at"),
    ("smallcnn_fgsm_at", "smallcnn", "fgsm_at"),
]
SEEDS = [42, 123, 2026]
EPSILONS = [0.0, 0.05, 0.2, 0.25]


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


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_fixture(raw_dir: Path, aggregated_dir: Path) -> None:
    model_summary_rows = [
        {"model": key, "architecture": arch, "training": training, "clean_accuracy_mean": 0.97}
        for key, arch, training in MODELS
    ]
    _write_csv(model_summary_rows, aggregated_dir / "model_summary.csv")

    robustness_rows = [
        {"model": key, "epsilon": eps, "robust_accuracy_mean": 0.9 - eps}
        for key, _, _ in MODELS
        for eps in EPSILONS
    ]
    _write_csv(robustness_rows, aggregated_dir / "robustness_summary.csv")

    fgsm_robustness_rows = [
        {"model": key, "seed": seed, "epsilon": eps, "robust_accuracy": 0.9 - eps}
        for key, _, _ in MODELS
        for seed in SEEDS
        for eps in EPSILONS
    ]
    _write_csv(fgsm_robustness_rows, raw_dir / "fgsm_robustness.csv")

    pgd10_rows = [
        {
            "model": key,
            "seed": seed,
            "epsilon": 0.25,
            "pgd_steps": 10,
            "pgd_random_start": True,
            "evaluated_samples": 100,
            "robust_accuracy": 0.1 + 0.01 * seed_idx,
        }
        for key, _, _ in MODELS
        for seed_idx, seed in enumerate(SEEDS)
    ]
    _write_csv(pgd10_rows, raw_dir / "pgd_whitebox.csv")

    pgd20_rows = [
        {
            "model": key,
            "seed": seed,
            "epsilon": 0.25,
            "pgd_steps": 20,
            "pgd_restarts": 5,
            "pgd_random_start": True,
            "evaluated_samples": 50,
            "robust_accuracy": 0.05 + 0.01 * seed_idx,
        }
        for key, _, _ in MODELS
        for seed_idx, seed in enumerate(SEEDS)
    ]
    _write_csv(pgd20_rows, raw_dir / "pgd20_restart5_whitebox.csv")
    _write_csv(
        [
            {"model": key, "robust_accuracy_mean": 0.06, "robust_accuracy_std": 0.02}
            for key, _, _ in MODELS
        ],
        aggregated_dir / "pgd20_restart5_summary.csv",
    )

    transfer_rows = [
        {
            "source_model": source,
            "target_model": target,
            "epsilon": 0.25,
            "conditional_transfer_success_rate_mean": 0.3 if source != target else 0.05,
        }
        for source, _, _ in MODELS
        for target, _, _ in MODELS
    ]
    _write_csv(transfer_rows, aggregated_dir / "transferability_summary.csv")

    # Detection AUC/TPR varies per seed (one row per model/seed/condition) so this
    # also exercises the seed-averaging path rather than a single-seed passthrough.
    detection_rows = [
        {
            "model": model,
            "seed": seed,
            "attack_condition": condition,
            "auc": 0.98 + 0.01 * seed_idx,
            "tpr_at_fpr_5": 0.9 + 0.05 * seed_idx,
        }
        for model in ("smallcnn_standard", "smallcnn_fgsm_at")
        for condition in ("FGSM", "PGD", "ALL_ATTACKS")
        for seed_idx, seed in enumerate(SEEDS)
    ]
    _write_csv(detection_rows, aggregated_dir / "firewall_detection_summary.csv")

    firewall_result_fields = {
        "evaluated_samples": 100,
        "threshold": 0.006,
        "min_confidence": 0.7,
        "purified_accuracy": 0.8,
        "detection_rate": 1.0,
        "reject_rate": 0.1,
        "accepted_accuracy": 0.85,
        "accept_original_rate": 0.6,
        "accept_purified_rate": 0.3,
        "reject_suspicious_rate": 0.1,
    }
    # original_accuracy/final_safe_accuracy vary per seed (one row per
    # model/seed/condition) so this exercises the seed-averaging path rather
    # than a single-seed passthrough (a prior bug silently kept only the last
    # seed's row per model/condition).
    firewall_results_rows = [
        {
            "model": model,
            "seed": seed,
            "condition": condition,
            "original_accuracy": 0.2 + 0.1 * seed_idx,
            "final_safe_accuracy": 0.7 + 0.1 * seed_idx,
            **firewall_result_fields,
        }
        for model in ("smallcnn_standard", "smallcnn_fgsm_at")
        for condition in ("Clean", "FGSM", "PGD")
        for seed_idx, seed in enumerate(SEEDS)
    ]
    _write_csv(firewall_results_rows, raw_dir / "firewall_results.csv")

    score_rows = [
        {"model": model, "condition": condition, "reconstruction_error": value, "threshold": 0.006}
        for model in ("smallcnn_standard", "smallcnn_fgsm_at")
        for condition, values in (
            ("Clean", [0.001, 0.002, 0.003]),
            ("FGSM", [0.02, 0.03, 0.04]),
            ("PGD", [0.015, 0.025, 0.035]),
        )
        for value in values
    ]
    _write_csv(score_rows, raw_dir / "firewall_detection_scores.csv")

    torch.save(
        {
            "model": "smallcnn_standard",
            "seed": 42,
            "epsilon": 0.25,
            "threshold": 0.006,
            "min_confidence": 0.7,
            "conditions": {
                "Clean": _condition_batch(),
                "FGSM": _condition_batch(),
                "PGD": _condition_batch(),
            },
        },
        raw_dir / "firewall_examples.pt",
    )


def test_build_presentation_from_saved_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    aggregated_dir = tmp_path / "aggregated"
    raw_dir.mkdir()
    aggregated_dir.mkdir()
    _build_fixture(raw_dir, aggregated_dir)

    config = {
        "seeds": SEEDS,
        "dataset": {"name": "MNIST"},
        "training": {"fgsm_adversarial_epsilon": 0.25},
        "paths": {
            "raw_dir": str(raw_dir),
            "aggregated_dir": str(aggregated_dir),
            "results_dir": str(tmp_path / "results"),
        },
    }

    output = build_presentation(config)

    assert output == tmp_path / "results" / "presentation" / "index.html"
    html = output.read_text(encoding="utf-8")
    assert "const DATA = " in html
    assert "Adversarial Firewall" in html
    assert "smallcnn_fgsm_at" in html
    assert html.count("data:image/png;base64,") == 18

    prefix = "const DATA = "
    start = html.index(prefix) + len(prefix)
    end = html.index(";\n</script>", start)
    data = json.loads(html[start:end])

    # Multi-seed rows must be averaged, not collapsed to whichever seed's row
    # happened to be iterated last (seed_idx 0, 1, 2 -> mean of 0.2/0.3/0.4 and
    # 0.7/0.8/0.9 respectively; see _build_fixture).
    pgd_result = data["firewall"]["results"]["smallcnn_fgsm_at"]["PGD"]
    assert pgd_result["original_accuracy"] == pytest.approx(0.3)
    assert pgd_result["final_safe_accuracy"] == pytest.approx(0.8)

    pgd_detection = data["firewall"]["detection"]["smallcnn_fgsm_at"]["PGD"]
    assert pgd_detection["auc"] == pytest.approx(0.99)
    assert pgd_detection["tprAtFpr5"] == pytest.approx(0.95)
