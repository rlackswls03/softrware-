"""Evaluate the Adversarial Firewall on SmallCNN classifiers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from adversarial_mnist.autoencoder import (
    ConvAutoencoder,
    autoencoder_checkpoint_path,
    load_autoencoder_checkpoint,
)
from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import load_trained_models, write_csv
from adversarial_mnist.firewall_evaluation import (
    build_detection_summary,
    calibrate_reconstruction_threshold,
    evaluate_firewall_model,
)
from adversarial_mnist.metrics import aggregate_mean_std
from adversarial_mnist.utils import ensure_dir, set_reproducibility, setup_logging
from scripts.common import add_common_args, prepare_config, prepare_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--pgd-steps", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--threshold-fpr", type=float, default=None)
    return parser


def _firewall_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("firewall", {})


def _selected_models(config: dict[str, Any], cli_models: list[str] | None) -> list[str]:
    if cli_models:
        return cli_models
    return list(_firewall_config(config).get("models", ["smallcnn_standard", "smallcnn_fgsm_at"]))


def _load_autoencoder(config: dict[str, Any], seed: int, device: torch.device) -> ConvAutoencoder:
    checkpoint = autoencoder_checkpoint_path(config["paths"]["checkpoints_dir"], seed)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing autoencoder checkpoint: {checkpoint}. "
            "Run `python -m scripts.train_autoencoder` first."
        )
    autoencoder = ConvAutoencoder().to(device)
    load_autoencoder_checkpoint(checkpoint, autoencoder, map_location=device)
    autoencoder.eval()
    return autoencoder


def _remove_firewall_outputs(config: dict[str, Any]) -> None:
    paths = [
        Path(config["paths"]["raw_dir"]) / "firewall_detection_scores.csv",
        Path(config["paths"]["raw_dir"]) / "firewall_results.csv",
        Path(config["paths"]["raw_dir"]) / "firewall_examples.pt",
        Path(config["paths"]["aggregated_dir"]) / "firewall_detection_summary.csv",
        Path(config["paths"]["aggregated_dir"]) / "firewall_results_summary.csv",
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def evaluate_firewall(
    config: dict[str, Any],
    device: torch.device,
    model_keys: list[str],
    force: bool = False,
    epsilon: float | None = None,
    pgd_steps: int | None = None,
    min_confidence: float | None = None,
    threshold_fpr: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate firewall metrics and write raw plus aggregated CSV files."""
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    aggregated_dir = ensure_dir(config["paths"]["aggregated_dir"])
    scores_path = raw_dir / "firewall_detection_scores.csv"
    results_path = raw_dir / "firewall_results.csv"
    examples_path = raw_dir / "firewall_examples.pt"
    if (scores_path.exists() or results_path.exists()) and not force:
        raise FileExistsError("Firewall result CSVs already exist. Use --force to overwrite.")
    if force:
        _remove_firewall_outputs(config)

    fw_config = _firewall_config(config)
    detector_config = fw_config.get("detector", {})
    attack_config = fw_config.get("attacks", {})
    threshold_fpr = float(
        detector_config.get("threshold_fpr", 0.05) if threshold_fpr is None else threshold_fpr
    )
    min_confidence = float(
        detector_config.get("min_confidence", 0.7)
        if min_confidence is None
        else min_confidence
    )
    epsilon = float(attack_config.get("epsilon", 0.25) if epsilon is None else epsilon)
    pgd_steps = int(attack_config.get("pgd_steps", 10) if pgd_steps is None else pgd_steps)
    pgd_alpha = attack_config.get("pgd_alpha")
    pgd_random_start = bool(attack_config.get("pgd_random_start", True))
    max_example_images = int(fw_config.get("max_example_images", 8))

    all_score_rows: list[dict[str, Any]] = []
    all_result_rows: list[dict[str, Any]] = []
    saved_examples: dict[str, Any] | None = None

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        set_reproducibility(seed, deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=seed)
        classifiers = load_trained_models(
            model_keys,
            config["paths"]["checkpoints_dir"],
            seed=seed,
            device=device,
        )
        autoencoder = _load_autoencoder(config, seed, device)
        threshold, validation_scores = calibrate_reconstruction_threshold(
            autoencoder,
            loaders.validation,
            device=device,
            fpr=threshold_fpr,
        )
        LOGGER.info(
            "Calibrated firewall threshold seed=%s threshold=%.6f from %s clean validation samples.",
            seed,
            threshold,
            int(validation_scores.numel()),
        )

        for model_key, classifier in classifiers.items():
            LOGGER.info(
                "Evaluating firewall model=%s seed=%s epsilon=%.3f PGD steps=%s.",
                model_key,
                seed,
                epsilon,
                pgd_steps,
            )
            score_rows, result_rows, examples = evaluate_firewall_model(
                model_key,
                classifier,
                autoencoder,
                loaders.test,
                seed=seed,
                device=device,
                threshold=threshold,
                threshold_fpr=threshold_fpr,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                pgd_alpha=pgd_alpha,
                pgd_random_start=pgd_random_start,
                min_confidence=min_confidence,
                max_example_images=max_example_images if saved_examples is None else 0,
            )
            all_score_rows.extend(score_rows)
            all_result_rows.extend(result_rows)
            if saved_examples is None and examples["conditions"]:
                saved_examples = examples

    scores = pd.DataFrame(all_score_rows)
    results = pd.DataFrame(all_result_rows)
    if scores.empty or results.empty:
        raise RuntimeError("Firewall evaluation produced no rows.")
    write_csv(all_score_rows, scores_path)
    write_csv(all_result_rows, results_path)
    if saved_examples is not None:
        torch.save(saved_examples, examples_path)

    detection_summary = build_detection_summary(scores, max_fpr=threshold_fpr)
    detection_summary.to_csv(aggregated_dir / "firewall_detection_summary.csv", index=False)
    aggregate_mean_std(
        results,
        group_columns=["model", "condition", "attack", "epsilon"],
        metric_columns=[
            "original_accuracy",
            "purified_accuracy",
            "detection_rate",
            "reject_rate",
            "accepted_accuracy",
            "final_safe_accuracy",
        ],
    ).to_csv(aggregated_dir / "firewall_results_summary.csv", index=False)
    LOGGER.info("Wrote firewall CSVs to %s and %s.", scores_path, results_path)
    return scores, results, detection_summary


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    device = prepare_device(args)
    evaluate_firewall(
        config,
        device,
        _selected_models(config, args.models),
        force=args.force,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        min_confidence=args.min_confidence,
        threshold_fpr=args.threshold_fpr,
    )


if __name__ == "__main__":
    main()
