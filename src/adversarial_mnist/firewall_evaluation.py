"""Evaluation routines for the Adversarial Firewall extension."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from adversarial_mnist.attacks import fgsm_attack, pgd_linf_attack
from adversarial_mnist.firewall import (
    ACCEPT_ORIGINAL,
    ACCEPT_PURIFIED,
    DECISION_LABELS,
    REJECT_SUSPICIOUS,
    autoencoder_reconstruction_error,
    binary_auc,
    choose_threshold_from_clean_validation,
    firewall_decisions,
    tpr_at_fpr,
)

FIREWALL_CONDITIONS = ("Clean", "FGSM", "PGD")


def predict_labels_confidences(
    model: nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted labels and max softmax confidences."""
    original_training = model.training
    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(inputs), dim=1)
        confidences, predictions = probabilities.max(dim=1)
    model.train(original_training)
    return predictions.detach(), confidences.detach()


def calibrate_reconstruction_threshold(
    autoencoder: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
    fpr: float = 0.05,
) -> tuple[float, torch.Tensor]:
    """Calibrate threshold from clean validation reconstruction scores."""
    score_batches: list[torch.Tensor] = []
    for inputs, _ in validation_loader:
        inputs = inputs.to(device)
        _, scores = autoencoder_reconstruction_error(autoencoder, inputs)
        score_batches.append(scores.cpu())
    if not score_batches:
        raise RuntimeError("Validation loader produced no batches for firewall calibration.")
    scores = torch.cat(score_batches)
    threshold = choose_threshold_from_clean_validation(scores, fpr=fpr)
    return threshold, scores


def _make_condition_inputs(
    condition: str,
    classifier: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    pgd_steps: int,
    pgd_alpha: float | None,
    pgd_random_start: bool,
) -> torch.Tensor:
    if condition == "Clean":
        return inputs.detach().clone()
    if condition == "FGSM":
        return fgsm_attack(classifier, inputs, targets, epsilon=epsilon)
    if condition == "PGD":
        return pgd_linf_attack(
            classifier,
            inputs,
            targets,
            epsilon=epsilon,
            steps=pgd_steps,
            alpha=pgd_alpha,
            random_start=pgd_random_start,
        )
    raise ValueError(f"Unknown firewall condition '{condition}'.")


def _final_prediction(
    decision: str,
    input_prediction: int,
    purified_prediction: int,
) -> int:
    if decision == ACCEPT_ORIGINAL:
        return input_prediction
    if decision == ACCEPT_PURIFIED:
        return purified_prediction
    if decision == REJECT_SUSPICIOUS:
        return -1
    raise ValueError(f"Unknown firewall decision '{decision}'.")


def _safe_count(condition: str, decision: str, final_prediction: int, target: int) -> int:
    if decision == REJECT_SUSPICIOUS:
        return 0 if condition == "Clean" else 1
    return int(final_prediction == target)


def _append_example_batch(
    examples: dict[str, Any],
    condition: str,
    max_examples: int,
    original_inputs: torch.Tensor,
    condition_inputs: torch.Tensor,
    purified_inputs: torch.Tensor,
    targets: torch.Tensor,
    clean_predictions: torch.Tensor,
    input_predictions: torch.Tensor,
    purified_predictions: torch.Tensor,
    scores: torch.Tensor,
    decisions: list[str],
) -> None:
    if max_examples <= 0 or condition in examples["conditions"]:
        return
    count = min(max_examples, int(targets.shape[0]))
    examples["conditions"][condition] = {
        "original": original_inputs[:count].detach().cpu(),
        "input": condition_inputs[:count].detach().cpu(),
        "purified": purified_inputs[:count].detach().cpu(),
        "labels": targets[:count].detach().cpu(),
        "clean_predictions": clean_predictions[:count].detach().cpu(),
        "input_predictions": input_predictions[:count].detach().cpu(),
        "purified_predictions": purified_predictions[:count].detach().cpu(),
        "scores": scores[:count].detach().cpu(),
        "decisions": decisions[:count],
    }


def evaluate_firewall_model(
    model_key: str,
    classifier: nn.Module,
    autoencoder: nn.Module,
    loader: DataLoader,
    seed: int,
    device: torch.device,
    threshold: float,
    threshold_fpr: float,
    epsilon: float = 0.25,
    pgd_steps: int = 10,
    pgd_alpha: float | None = None,
    pgd_random_start: bool = True,
    min_confidence: float = 0.7,
    max_example_images: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate detector, purifier, and reject policy for one classifier."""
    classifier.eval()
    autoencoder.eval()
    score_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    examples: dict[str, Any] = {
        "model": model_key,
        "seed": seed,
        "epsilon": epsilon,
        "threshold": threshold,
        "min_confidence": min_confidence,
        "conditions": {},
    }

    for condition in FIREWALL_CONDITIONS:
        stats = Counter(
            {
                "total": 0,
                "original_correct": 0,
                "purified_correct": 0,
                "detected": 0,
                "accepted": 0,
                "accepted_correct": 0,
                "safe": 0,
            }
        )
        decision_counts = Counter({label: 0 for label in DECISION_LABELS})
        sample_offset = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            condition_inputs = _make_condition_inputs(
                condition,
                classifier,
                inputs,
                targets,
                epsilon=epsilon,
                pgd_steps=pgd_steps,
                pgd_alpha=pgd_alpha,
                pgd_random_start=pgd_random_start,
            )
            purified_inputs, scores = autoencoder_reconstruction_error(autoencoder, condition_inputs)
            clean_predictions, clean_confidences = predict_labels_confidences(classifier, inputs)
            input_predictions, input_confidences = predict_labels_confidences(
                classifier, condition_inputs
            )
            purified_predictions, purified_confidences = predict_labels_confidences(
                classifier, purified_inputs
            )
            decisions = firewall_decisions(
                scores,
                threshold,
                purified_confidences,
                min_confidence=min_confidence,
            )
            detected = scores > threshold
            original_correct = input_predictions == targets
            purified_correct = purified_predictions == targets

            _append_example_batch(
                examples,
                condition,
                max_example_images,
                inputs,
                condition_inputs,
                purified_inputs,
                targets,
                clean_predictions,
                input_predictions,
                purified_predictions,
                scores,
                decisions,
            )

            for batch_index, decision in enumerate(decisions):
                target = int(targets[batch_index].item())
                input_prediction = int(input_predictions[batch_index].item())
                purified_prediction = int(purified_predictions[batch_index].item())
                final_prediction = _final_prediction(decision, input_prediction, purified_prediction)
                final_correct = final_prediction == target if final_prediction >= 0 else False
                safe = _safe_count(condition, decision, final_prediction, target)

                stats["total"] += 1
                stats["original_correct"] += int(original_correct[batch_index].item())
                stats["purified_correct"] += int(purified_correct[batch_index].item())
                stats["detected"] += int(detected[batch_index].item())
                stats["accepted"] += int(decision != REJECT_SUSPICIOUS)
                stats["accepted_correct"] += int(final_correct)
                stats["safe"] += safe
                decision_counts[decision] += 1

                score_rows.append(
                    {
                        "model": model_key,
                        "seed": seed,
                        "condition": condition,
                        "attack": "NONE" if condition == "Clean" else condition,
                        "epsilon": 0.0 if condition == "Clean" else float(epsilon),
                        "sample_index": sample_offset + batch_index,
                        "true_label": target,
                        "clean_prediction": int(clean_predictions[batch_index].item()),
                        "clean_confidence": float(clean_confidences[batch_index].item()),
                        "input_prediction": input_prediction,
                        "input_confidence": float(input_confidences[batch_index].item()),
                        "purified_prediction": purified_prediction,
                        "purified_confidence": float(purified_confidences[batch_index].item()),
                        "reconstruction_error": float(scores[batch_index].item()),
                        "threshold": float(threshold),
                        "threshold_fpr": float(threshold_fpr),
                        "detected": bool(detected[batch_index].item()),
                        "decision": decision,
                        "final_prediction": final_prediction,
                        "original_correct": bool(original_correct[batch_index].item()),
                        "purified_correct": bool(purified_correct[batch_index].item()),
                        "final_correct": bool(final_correct),
                        "safe_outcome": bool(safe),
                        "is_attack": condition != "Clean",
                        "device": str(device),
                    }
                )
            sample_offset += int(targets.shape[0])

        total = int(stats["total"])
        if total == 0:
            raise RuntimeError("Firewall evaluation loader produced zero samples.")
        accepted = int(stats["accepted"])
        result_rows.append(
            {
                "model": model_key,
                "seed": seed,
                "condition": condition,
                "attack": "NONE" if condition == "Clean" else condition,
                "epsilon": 0.0 if condition == "Clean" else float(epsilon),
                "evaluated_samples": total,
                "threshold": float(threshold),
                "threshold_fpr": float(threshold_fpr),
                "min_confidence": float(min_confidence),
                "original_accuracy": stats["original_correct"] / total,
                "purified_accuracy": stats["purified_correct"] / total,
                "detection_rate": stats["detected"] / total,
                "false_positive_rate": stats["detected"] / total
                if condition == "Clean"
                else math.nan,
                "reject_rate": decision_counts[REJECT_SUSPICIOUS] / total,
                "accepted_accuracy": stats["accepted_correct"] / accepted
                if accepted > 0
                else math.nan,
                "final_safe_accuracy": stats["safe"] / total,
                "accept_original_rate": decision_counts[ACCEPT_ORIGINAL] / total,
                "accept_purified_rate": decision_counts[ACCEPT_PURIFIED] / total,
                "reject_suspicious_rate": decision_counts[REJECT_SUSPICIOUS] / total,
                "device": str(device),
            }
        )

    return score_rows, result_rows, examples


def build_detection_summary(scores: pd.DataFrame, max_fpr: float = 0.05) -> pd.DataFrame:
    """Create AUC and TPR@FPR summary rows from firewall detection scores."""
    required = {"model", "seed", "condition", "reconstruction_error", "is_attack"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"firewall scores are missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for (model_key, seed), group in scores.groupby(["model", "seed"], sort=False):
        clean = group[group["condition"] == "Clean"]
        for attack_condition in ("FGSM", "PGD", "ALL_ATTACKS"):
            if attack_condition == "ALL_ATTACKS":
                attack = group[group["condition"].isin(["FGSM", "PGD"])]
            else:
                attack = group[group["condition"] == attack_condition]
            combined = pd.concat([clean, attack], ignore_index=True)
            if clean.empty or attack.empty:
                continue
            labels = combined["is_attack"].astype(int).to_numpy()
            score_values = combined["reconstruction_error"].astype(float).to_numpy()
            rows.append(
                {
                    "model": model_key,
                    "seed": int(seed),
                    "attack_condition": attack_condition,
                    "auc": binary_auc(score_values, labels),
                    "tpr_at_fpr_5": tpr_at_fpr(score_values, labels, max_fpr=max_fpr),
                    "clean_samples": int(clean.shape[0]),
                    "attack_samples": int(attack.shape[0]),
                }
            )
    return pd.DataFrame(rows)
