"""Build a presentation-style story deck from saved experiment artifacts.

Unlike `simulation.py` (a single-page inspection tool) this module produces a
slide-by-slide narrative deck for live presentation: FGSM adversarial training's
apparent win, its PGD instability, transfer-attack asymmetry, and the
Adversarial Firewall recovery, ending in an interactive pipeline walkthrough.
All numbers are read from saved CSV/JSON/checkpoint artifacts; nothing is
recomputed and nothing is hand-authored.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adversarial_mnist.metrics import aggregate_mean_std, clean_accuracy_retention
from adversarial_mnist.models import count_parameters, create_model
from adversarial_mnist.presentation_html import render_presentation_html
from adversarial_mnist.simulation import _build_examples_payload, _require_file, _torch_load
from adversarial_mnist.utils import ensure_dir

MODEL_ORDER: tuple[str, ...] = (
    "lenet_standard",
    "smallcnn_standard",
    "lenet_fgsm_at",
    "smallcnn_fgsm_at",
)

MODEL_LABELS: dict[str, str] = {
    "lenet_standard": "LeNet · 표준 학습",
    "smallcnn_standard": "SmallCNN · 표준 학습",
    "lenet_fgsm_at": "LeNet · FGSM 훈련",
    "smallcnn_fgsm_at": "SmallCNN · FGSM 훈련",
}

MODEL_SHORT_LABELS: dict[str, str] = {
    "lenet_standard": "LeNet 표준",
    "smallcnn_standard": "SmallCNN 표준",
    "lenet_fgsm_at": "LeNet FGSM-AT",
    "smallcnn_fgsm_at": "SmallCNN FGSM-AT",
}

# Categorical slots chosen from the validated dark-mode palette, deliberately
# avoiding green/red/amber (reserved for accept/reject status colors below).
MODEL_COLORS: dict[str, str] = {
    "lenet_standard": "#3987e5",
    "smallcnn_standard": "#9085e9",
    "lenet_fgsm_at": "#c98500",
    "smallcnn_fgsm_at": "#d55181",
}

CONDITION_LABELS: dict[str, str] = {
    "Clean": "정상 입력",
    "FGSM": "FGSM 공격",
    "PGD": "PGD 공격",
}

SCORE_HIST_BINS = 40

# Structural facts about the two architectures (verified against models.py);
# only paramCount is computed live to avoid a stale hand-typed number.
ARCHITECTURE_FACTS: dict[str, dict[str, str]] = {
    "lenet": {
        "label": "LeNet",
        "convLayers": "2",
        "filters": "6, 16",
        "kernel": "5×5",
        "pooling": "AvgPool",
        "batchNorm": "없음",
        "dropout": "없음",
    },
    "smallcnn": {
        "label": "SmallCNN",
        "convLayers": "4",
        "filters": "32, 32, 64, 64",
        "kernel": "3×3",
        "pooling": "MaxPool",
        "batchNorm": "1, 3번째 Conv 후",
        "dropout": "FC 앞 각각 p=0.25",
    },
}


def _architecture_meta() -> dict[str, Any]:
    meta = {}
    for architecture, facts in ARCHITECTURE_FACTS.items():
        params = count_parameters(create_model(architecture))
        meta[architecture] = {**facts, "paramCount": int(params)}
    return meta


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required presentation artifact is missing: {path}")
    return pd.read_csv(path)


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _clean_value(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _model_meta(aggregated_dir: Path) -> list[dict[str, Any]]:
    summary = _read_csv(aggregated_dir / "model_summary.csv")
    by_model = {row["model"]: row for _, row in summary.iterrows()}
    meta = []
    for model in MODEL_ORDER:
        if model not in by_model:
            continue
        row = by_model[model]
        meta.append(
            {
                "model": model,
                "label": MODEL_LABELS[model],
                "shortLabel": MODEL_SHORT_LABELS[model],
                "color": MODEL_COLORS[model],
                "architecture": row["architecture"],
                "training": row["training"],
                "cleanAccuracy": float(row["clean_accuracy_mean"]),
            }
        )
    return meta


def _clean_retention(aggregated_dir: Path) -> list[dict[str, Any]]:
    summary = _read_csv(aggregated_dir / "model_summary.csv")
    by_key = {
        (row["architecture"], row["training"]): float(row["clean_accuracy_mean"])
        for _, row in summary.iterrows()
    }
    retention = []
    for architecture in ("lenet", "smallcnn"):
        standard = by_key.get((architecture, "standard"))
        defended = by_key.get((architecture, "fgsm_at"))
        if standard is None or defended is None:
            continue
        retention.append(
            {
                "architecture": architecture,
                "standardAccuracy": standard,
                "defendedAccuracy": defended,
                "retentionPct": clean_accuracy_retention(defended, standard),
            }
        )
    return retention


def _fgsm_curve(aggregated_dir: Path) -> dict[str, Any]:
    summary = _read_csv(aggregated_dir / "robustness_summary.csv")
    epsilons = sorted(summary["epsilon"].unique().tolist())
    series: dict[str, list[float]] = {}
    for model in MODEL_ORDER:
        model_rows = summary[summary["model"] == model].set_index("epsilon")
        series[model] = [
            float(model_rows.loc[eps, "robust_accuracy_mean"]) if eps in model_rows.index else None
            for eps in epsilons
        ]
    return {"epsilons": epsilons, "series": series}


def _seed_points(raw: pd.DataFrame, model: str, value_column: str) -> list[dict[str, Any]]:
    rows = raw[raw["model"] == model].sort_values("seed")
    return [
        {"seed": int(row["seed"]), "value": float(row[value_column])} for _, row in rows.iterrows()
    ]


def _pgd_panel(raw_dir: Path, mean_std: pd.DataFrame, value_column: str) -> dict[str, Any]:
    panel: dict[str, Any] = {}
    mean_col = f"{value_column}_mean"
    std_col = f"{value_column}_std"
    by_model = mean_std.set_index("model")
    for model in MODEL_ORDER:
        if model not in by_model.index:
            continue
        row = by_model.loc[model]
        panel[model] = {
            "mean": float(row[mean_col]),
            "std": float(row[std_col]) if not pd.isna(row[std_col]) else 0.0,
            "seeds": _seed_points(raw_dir, model, value_column),
        }
    return panel


def _pgd10_panel(raw_dir: Path) -> dict[str, Any]:
    raw = _read_csv(raw_dir / "pgd_whitebox.csv")
    mean_std = aggregate_mean_std(raw, group_columns=["model"], metric_columns=["robust_accuracy"])
    return _pgd_panel(raw, mean_std, "robust_accuracy")


def _pgd20_panel(raw_dir: Path, aggregated_dir: Path) -> dict[str, Any]:
    raw = _read_csv(raw_dir / "pgd20_restart5_whitebox.csv")
    mean_std = _read_csv(aggregated_dir / "pgd20_restart5_summary.csv")
    return _pgd_panel(raw, mean_std, "robust_accuracy")


def _transfer_matrix(aggregated_dir: Path, epsilon: float = 0.25) -> dict[str, Any]:
    summary = _read_csv(aggregated_dir / "transferability_summary.csv")
    subset = summary[np.isclose(summary["epsilon"], epsilon)]
    lookup = {
        (row["source_model"], row["target_model"]): float(
            row["conditional_transfer_success_rate_mean"]
        )
        for _, row in subset.iterrows()
    }
    matrix = [
        [lookup.get((source, target)) for target in MODEL_ORDER] for source in MODEL_ORDER
    ]
    return {
        "epsilon": epsilon,
        "models": list(MODEL_ORDER),
        "labels": [MODEL_SHORT_LABELS[m] for m in MODEL_ORDER],
        "matrix": matrix,
    }


def _fgsm_seed_rows(raw_dir: Path, model: str, seed: int) -> list[dict[str, Any]]:
    raw = _read_csv(raw_dir / "fgsm_robustness.csv")
    rows = raw[(raw["model"] == model) & (raw["seed"] == seed)].sort_values("epsilon")
    return [
        {"epsilon": float(row["epsilon"]), "robustAccuracy": float(row["robust_accuracy"])}
        for _, row in rows.iterrows()
    ]


def _firewall_detection(aggregated_dir: Path) -> dict[str, Any]:
    summary = _read_csv(aggregated_dir / "firewall_detection_summary.csv")
    detection: dict[str, Any] = {}
    for model, group in summary.groupby("model"):
        detection[model] = {
            row["attack_condition"]: {
                "auc": float(row["auc"]),
                "tprAtFpr5": float(row["tpr_at_fpr_5"]),
            }
            for _, row in group.iterrows()
        }
    return detection


FIREWALL_RESULT_FIELDS = (
    "evaluated_samples",
    "threshold",
    "min_confidence",
    "original_accuracy",
    "purified_accuracy",
    "detection_rate",
    "reject_rate",
    "accepted_accuracy",
    "final_safe_accuracy",
    "accept_original_rate",
    "accept_purified_rate",
    "reject_suspicious_rate",
)


def _firewall_results(results_path: Path) -> dict[str, Any]:
    raw = _read_csv(results_path)
    results: dict[str, Any] = {}
    for model, group in raw.groupby("model"):
        results[model] = {
            row["condition"]: {field: _clean_value(row[field]) for field in FIREWALL_RESULT_FIELDS}
            for _, row in group.iterrows()
        }
    return results


def _score_histograms(raw_dir: Path) -> dict[str, Any]:
    raw = _read_csv(raw_dir / "firewall_detection_scores.csv")
    upper_bound = float(np.ceil(raw["reconstruction_error"].max() * 1000.0) / 1000.0 + 0.002)
    bin_edges = np.linspace(0.0, upper_bound, SCORE_HIST_BINS + 1)
    histograms: dict[str, Any] = {}
    for (model, condition), group in raw.groupby(["model", "condition"]):
        counts, _ = np.histogram(group["reconstruction_error"].to_numpy(), bins=bin_edges)
        histograms.setdefault(model, {})[condition] = counts.astype(int).tolist()
    return {
        "binEdges": bin_edges.tolist(),
        "threshold": float(raw["threshold"].iloc[0]),
        "series": histograms,
    }


def _references() -> list[dict[str, str]]:
    return [
        {"id": 1, "text": "Goodfellow, Shlens, Szegedy — Explaining and Harnessing Adversarial Examples (2014)"},
        {"id": 2, "text": "Papernot, McDaniel, Goodfellow — Transferability in Machine Learning (2016)"},
        {"id": 3, "text": "Kurakin, Goodfellow, Bengio — Adversarial Machine Learning at Scale (2016)"},
        {"id": 4, "text": "Madry et al. — Towards Deep Learning Models Resistant to Adversarial Attacks (2017)"},
        {"id": 5, "text": "Wong, Rice, Kolter — Fast is better than free: Revisiting adversarial training (2020)"},
        {"id": 6, "text": "Athalye, Carlini, Wagner — Obfuscated Gradients Give a False Sense of Security (2018)"},
    ]


def _experiment_config(config: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    pgd10_row = _read_csv(raw_dir / "pgd_whitebox.csv").iloc[0]
    pgd20_row = _read_csv(raw_dir / "pgd20_restart5_whitebox.csv").iloc[0]
    return {
        "dataset": config.get("dataset", {}).get("name", "MNIST"),
        "fgsmAtEpsilon": float(config["training"]["fgsm_adversarial_epsilon"]),
        "pgd10": {
            "epsilon": float(pgd10_row["epsilon"]),
            "steps": int(pgd10_row["pgd_steps"]),
            "randomStart": bool(pgd10_row["pgd_random_start"]),
            "evaluatedSamples": int(pgd10_row["evaluated_samples"]),
        },
        "pgd20": {
            "epsilon": float(pgd20_row["epsilon"]),
            "steps": int(pgd20_row["pgd_steps"]),
            "restarts": int(pgd20_row["pgd_restarts"]),
            "randomStart": bool(pgd20_row["pgd_random_start"]),
            "evaluatedSamples": int(pgd20_row["evaluated_samples"]),
        },
    }


def _build_payload(config: dict[str, Any], examples_path: Path, results_path: Path) -> dict[str, Any]:
    raw_dir = Path(config["paths"]["raw_dir"])
    aggregated_dir = Path(config["paths"]["aggregated_dir"])

    pgd10 = _pgd10_panel(raw_dir)
    pgd20 = _pgd20_panel(raw_dir, aggregated_dir)
    examples = _build_examples_payload(_torch_load(examples_path))

    payload = {
        "seeds": [int(seed) for seed in config["seeds"]],
        "experimentConfig": _experiment_config(config, raw_dir),
        "models": _model_meta(aggregated_dir),
        "architectures": _architecture_meta(),
        "cleanRetention": _clean_retention(aggregated_dir),
        "fgsmCurve": _fgsm_curve(aggregated_dir),
        "fgsmNonMonotonic": {
            "lenet_fgsm_at_seed42": _fgsm_seed_rows(raw_dir, "lenet_fgsm_at", 42),
            "smallcnn_fgsm_at_seed2026": _fgsm_seed_rows(raw_dir, "smallcnn_fgsm_at", 2026),
        },
        "pgd": {"pgd10": pgd10, "pgd20": pgd20},
        "transfer": _transfer_matrix(aggregated_dir),
        "firewall": {
            "detection": _firewall_detection(aggregated_dir),
            "results": _firewall_results(results_path),
            "scoreHist": _score_histograms(raw_dir),
            "examples": examples,
        },
        "references": _references(),
        "sourceFiles": {
            "examples": str(examples_path),
            "results": str(results_path),
        },
    }
    return payload


def build_presentation(
    config: dict[str, Any],
    examples_path: str | Path | None = None,
    results_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a standalone interactive presentation deck from saved results."""
    raw_dir = Path(config["paths"]["raw_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    examples_file = _require_file(examples_path or raw_dir / "firewall_examples.pt")
    results_file = _require_file(results_path or raw_dir / "firewall_results.csv")
    output_root = ensure_dir(output_dir or results_dir / "presentation")

    payload = _build_payload(config, examples_file, results_file)
    output_path = output_root / "index.html"
    output_path.write_text(render_presentation_html(payload), encoding="utf-8")
    return output_path
