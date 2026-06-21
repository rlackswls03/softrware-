"""Matplotlib visualizations and Markdown summary generation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib
import numpy as np
import pandas as pd
import torch

from adversarial_mnist.metrics import clean_accuracy_retention
from adversarial_mnist.models import MODEL_ORDER
from adversarial_mnist.utils import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _require_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Required result file is missing: {target}")
    return target


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str | Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def _format_percent_from_100(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{value:.2f}%"


def plot_robustness_curve(
    robustness_csv: str | Path,
    output_path: str | Path,
) -> None:
    """Plot FGSM epsilon robustness curves with seed mean/std."""
    source = _require_file(robustness_csv)
    frame = pd.read_csv(source)
    _require_columns(frame, {"model", "epsilon", "robust_accuracy"}, source)
    output = Path(output_path)
    ensure_dir(output.parent)

    plt.figure(figsize=(9, 6))
    for model_key, group in frame.groupby("model", sort=False):
        stats = (
            group.groupby("epsilon")["robust_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("epsilon")
        )
        std = stats["std"].fillna(0.0)
        plt.plot(stats["epsilon"], stats["mean"], marker="o", label=model_key)
        plt.fill_between(
            stats["epsilon"],
            stats["mean"] - std,
            stats["mean"] + std,
            alpha=0.15,
        )
    plt.title("FGSM Robustness Curve")
    plt.xlabel("FGSM epsilon in [0, 1] pixel space")
    plt.ylabel("Robust accuracy")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_transferability_heatmap(
    transfer_csv: str | Path,
    output_path: str | Path,
    epsilon: float = 0.25,
    metric: str = "conditional_transfer_success_rate",
) -> None:
    """Plot a 4x4 transferability heatmap for one epsilon."""
    source = _require_file(transfer_csv)
    frame = pd.read_csv(source)
    _require_columns(frame, {"source_model", "target_model", "epsilon", metric}, source)
    selected = frame[np.isclose(frame["epsilon"].astype(float), epsilon)]
    if selected.empty:
        raise ValueError(f"No transferability rows found for epsilon={epsilon}.")
    pivot = (
        selected.groupby(["source_model", "target_model"])[metric]
        .mean()
        .unstack()
        .reindex(index=MODEL_ORDER, columns=MODEL_ORDER)
    )

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(8, 7))
    values = pivot.to_numpy(dtype=float)
    image = plt.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04, label=metric)
    plt.xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=35, ha="right")
    plt.yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    plt.xlabel("Target model")
    plt.ylabel("Source model")
    plt.title(f"FGSM Transferability Heatmap (epsilon={epsilon:g})")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = "NaN" if math.isnan(value) else f"{value:.2f}"
            plt.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if not math.isnan(value) and value > 0.5 else "black",
            )
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_all_transferability_heatmaps(
    transfer_csv: str | Path,
    figures_dir: str | Path,
    epsilons: list[float],
) -> None:
    """Plot one transferability heatmap for every configured epsilon."""
    for epsilon in epsilons:
        plot_transferability_heatmap(
            transfer_csv,
            Path(figures_dir) / f"transferability_eps_{epsilon:.2f}.png",
            epsilon=epsilon,
        )


def plot_clean_robust_comparison(
    clean_csv: str | Path,
    robustness_csv: str | Path,
    output_path: str | Path,
    epsilon: float = 0.25,
) -> None:
    """Plot clean accuracy against FGSM robust accuracy at a target epsilon."""
    clean_source = _require_file(clean_csv)
    robust_source = _require_file(robustness_csv)
    clean = pd.read_csv(clean_source)
    robust = pd.read_csv(robust_source)
    _require_columns(clean, {"model", "clean_accuracy"}, clean_source)
    _require_columns(robust, {"model", "epsilon", "robust_accuracy"}, robust_source)
    clean_mean = clean.groupby("model")["clean_accuracy"].mean()
    robust_mean = robust[np.isclose(robust["epsilon"].astype(float), epsilon)].groupby("model")[
        "robust_accuracy"
    ].mean()

    models = [model for model in MODEL_ORDER if model in clean_mean.index]
    x_positions = np.arange(len(models))
    width = 0.38
    output = Path(output_path)
    ensure_dir(output.parent)

    plt.figure(figsize=(10, 6))
    plt.bar(x_positions - width / 2, [clean_mean[m] for m in models], width, label="Clean")
    plt.bar(
        x_positions + width / 2,
        [robust_mean.get(m, np.nan) for m in models],
        width,
        label=f"FGSM robust eps={epsilon:g}",
    )
    plt.xticks(x_positions, models, rotation=25, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.02)
    plt.title("Clean vs Robust Accuracy")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_clean_accuracy_retention(
    clean_csv: str | Path,
    output_path: str | Path,
) -> None:
    """Plot clean accuracy retention for FGSM adversarially trained models."""
    clean_source = _require_file(clean_csv)
    clean = pd.read_csv(clean_source)
    _require_columns(clean, {"model", "clean_accuracy"}, clean_source)
    clean_means = clean.groupby("model")["clean_accuracy"].mean()
    retention = _retention_by_model(clean_means.reindex(MODEL_ORDER))
    models = ["lenet_fgsm_at", "smallcnn_fgsm_at"]
    values = [retention.get(model_key, math.nan) for model_key in models]

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, values, color=["#4c78a8", "#59a14f"])
    plt.axhline(100.0, color="#444444", linestyle="--", linewidth=1, label="Standard counterpart")
    plt.ylabel("Clean accuracy retention (%)")
    plt.title("Clean Accuracy Retention of FGSM-AT Models")
    upper = max([100.0, *[value for value in values if not math.isnan(value)]]) * 1.15
    plt.ylim(0.0, upper)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for bar, value in zip(bars, values, strict=True):
        label = "NaN" if math.isnan(value) else f"{value:.1f}%"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            0.0 if math.isnan(value) else value,
            label,
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_pgd_whitebox_bar(
    pgd_csv: str | Path,
    output_path: str | Path,
) -> None:
    """Plot PGD white-box robust accuracy with seed points and std error bars."""
    pgd_source = _require_file(pgd_csv)
    pgd = pd.read_csv(pgd_source)
    _require_columns(pgd, {"model", "seed", "robust_accuracy", "epsilon", "pgd_steps"}, pgd_source)
    stats = pgd.groupby("model")["robust_accuracy"].agg(["mean", "std"]).reindex(MODEL_ORDER)
    models = [model_key for model_key in MODEL_ORDER if model_key in stats.index]
    means = [float(stats.loc[model_key, "mean"]) for model_key in models]
    stds = [float(stats.loc[model_key, "std"]) if not pd.isna(stats.loc[model_key, "std"]) else 0.0 for model_key in models]
    epsilon = float(pgd["epsilon"].iloc[0])
    steps = int(pgd["pgd_steps"].iloc[0])
    x_positions = np.arange(len(models))

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(10, 6.2))
    bars = plt.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=6,
        color="#8e6c8a",
        alpha=0.8,
        label="mean ± std",
    )
    seeds = sorted(pgd["seed"].unique())
    offsets = np.linspace(-0.18, 0.18, num=len(seeds)) if len(seeds) > 1 else np.array([0.0])
    for offset, seed in zip(offsets, seeds, strict=True):
        seed_rows = pgd[pgd["seed"] == seed].set_index("model")
        values = [float(seed_rows.loc[model_key, "robust_accuracy"]) for model_key in models]
        plt.scatter(
            x_positions + offset,
            values,
            s=46,
            edgecolors="black",
            linewidths=0.5,
            label=f"seed {seed}",
            zorder=3,
        )
    plt.ylabel("Robust accuracy")
    upper = max(0.45, max(mean + std for mean, std in zip(means, stds, strict=True)) + 0.08)
    plt.ylim(0.0, min(1.02, upper))
    plt.title(f"PGD White-Box Robust Accuracy (epsilon={epsilon:g}, steps={steps})")
    plt.xticks(x_positions, models, rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=2)
    for bar, mean, std in zip(bars, means, stds, strict=True):
        label = f"{mean * 100:.1f}% ± {std * 100:.1f}p"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_adversarial_examples(
    examples_path: str | Path,
    output_path: str | Path,
) -> None:
    """Plot original, FGSM, and PGD example images from a saved tensor batch."""
    source = _require_file(examples_path)
    examples = torch.load(source, map_location="cpu")
    required = {"original", "fgsm", "pgd", "labels"}
    missing = required - set(examples)
    if missing:
        raise ValueError(f"{source} is missing keys: {sorted(missing)}")
    original = examples["original"]
    fgsm = examples["fgsm"]
    pgd = examples["pgd"]
    labels = examples["labels"]
    columns = min(8, int(original.shape[0]))
    rows = [("Original", original), ("FGSM", fgsm), ("PGD", pgd)]

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(columns * 1.4, 4.8))
    for row_index, (row_name, tensor) in enumerate(rows):
        for column_index in range(columns):
            axis = plt.subplot(len(rows), columns, row_index * columns + column_index + 1)
            axis.imshow(tensor[column_index, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(f"y={int(labels[column_index])}", fontsize=9)
            if column_index == 0:
                axis.set_ylabel(row_name, fontsize=10)
    epsilon = examples.get("epsilon")
    title = "Original / FGSM / PGD examples"
    if epsilon is not None:
        title += f" (epsilon={float(epsilon):g})"
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_all_results(config: dict[str, Any]) -> None:
    """Create all figures from saved result artifacts."""
    raw_dir = Path(config["paths"]["raw_dir"])
    figures_dir = ensure_dir(config["paths"]["figures_dir"])
    epsilon = float(config["evaluation"]["pgd"].get("epsilon", 0.25))
    plot_robustness_curve(raw_dir / "fgsm_robustness.csv", figures_dir / "robustness_curve.png")
    plot_all_transferability_heatmaps(
        raw_dir / "transferability_long.csv",
        figures_dir,
        [float(value) for value in config["evaluation"]["fgsm_epsilons"]],
    )
    plot_clean_robust_comparison(
        raw_dir / "clean_accuracy.csv",
        raw_dir / "fgsm_robustness.csv",
        figures_dir / "clean_robust_comparison.png",
        epsilon=epsilon,
    )
    plot_clean_accuracy_retention(
        raw_dir / "clean_accuracy.csv",
        figures_dir / "clean_accuracy_retention.png",
    )
    plot_pgd_whitebox_bar(raw_dir / "pgd_whitebox.csv", figures_dir / "pgd_whitebox.png")
    plot_adversarial_examples(
        raw_dir / "adversarial_examples.pt",
        figures_dir / "adversarial_examples.png",
    )
    generate_figures_index(figures_dir, figures_dir / "figure_index.md")


def _model_clean_means(clean: pd.DataFrame) -> pd.Series:
    return clean.groupby("model")["clean_accuracy"].mean().reindex(MODEL_ORDER)


def _robust_at_epsilon(robust: pd.DataFrame, epsilon: float) -> pd.Series:
    selected = robust[np.isclose(robust["epsilon"].astype(float), epsilon)]
    return selected.groupby("model")["robust_accuracy"].mean().reindex(MODEL_ORDER)


def _retention_by_model(clean_means: pd.Series) -> dict[str, float]:
    return {
        "lenet_fgsm_at": clean_accuracy_retention(
            float(clean_means.get("lenet_fgsm_at", math.nan)),
            float(clean_means.get("lenet_standard", math.nan)),
        ),
        "smallcnn_fgsm_at": clean_accuracy_retention(
            float(clean_means.get("smallcnn_fgsm_at", math.nan)),
            float(clean_means.get("smallcnn_standard", math.nan)),
        ),
    }


def _transfer_direction_means(transfer: pd.DataFrame, epsilon: float) -> tuple[pd.Series, pd.Series]:
    selected = transfer[
        np.isclose(transfer["epsilon"].astype(float), epsilon)
        & (transfer["source_model"] != transfer["target_model"])
    ]
    outgoing = selected.groupby("source_model")["conditional_transfer_success_rate"].mean()
    incoming = selected.groupby("target_model")["conditional_transfer_success_rate"].mean()
    return outgoing.reindex(MODEL_ORDER), incoming.reindex(MODEL_ORDER)


def generate_summary_markdown(config: dict[str, Any], output_path: str | Path) -> None:
    """Generate ``results/summary.md`` from actual CSV results."""
    raw_dir = Path(config["paths"]["raw_dir"])
    clean_file = _require_file(raw_dir / "clean_accuracy.csv")
    robust_file = _require_file(raw_dir / "fgsm_robustness.csv")
    transfer_file = _require_file(raw_dir / "transferability_long.csv")
    pgd_file = _require_file(raw_dir / "pgd_whitebox.csv")

    clean = pd.read_csv(clean_file)
    robust = pd.read_csv(robust_file)
    transfer = pd.read_csv(transfer_file)
    pgd = pd.read_csv(pgd_file)
    epsilon = float(config["evaluation"]["pgd"].get("epsilon", 0.25))
    reference_target = float(config.get("reference_robust_accuracy_target", 0.9))

    clean_means = _model_clean_means(clean)
    robust_means = _robust_at_epsilon(robust, epsilon)
    retention = _retention_by_model(clean_means)
    outgoing, incoming = _transfer_direction_means(transfer, epsilon)
    pgd_means = pgd.groupby("model")["robust_accuracy"].mean().reindex(MODEL_ORDER)

    lines: list[str] = [
        "# Experiment Summary",
        "",
    ]
    if config.get("quick", {}).get("enabled"):
        lines.extend(
            [
                "> WARNING: quick mode result. This run is for pipeline connectivity only; "
                "do not draw performance conclusions.",
                "",
            ]
        )

    lines.extend(
        [
            "## Clean Accuracy and FGSM Robustness",
            "",
            f"| Model | Clean accuracy | FGSM eps={epsilon:g} robust accuracy | Clean accuracy retention | Meets 90% reference target? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for model_key in MODEL_ORDER:
        retention_value = retention.get(model_key, math.nan)
        robust_value = float(robust_means.get(model_key, math.nan))
        meets_target = "N/A"
        if not math.isnan(robust_value):
            meets_target = "yes" if robust_value >= reference_target else "no"
        lines.append(
            "| "
            f"{model_key} | "
            f"{_format_percent(float(clean_means.get(model_key, math.nan)))} | "
            f"{_format_percent(robust_value)} | "
            f"{_format_percent_from_100(retention_value)} | "
            f"{meets_target} |"
        )

    lines.extend(
        [
            "",
            "## Transferability",
            "",
            "| Model | Mean outgoing conditional transfer success | Mean incoming conditional transfer success |",
            "|---|---:|---:|",
        ]
    )
    for model_key in MODEL_ORDER:
        lines.append(
            "| "
            f"{model_key} | "
            f"{_format_percent(float(outgoing.get(model_key, math.nan)))} | "
            f"{_format_percent(float(incoming.get(model_key, math.nan)))} |"
        )

    lines.extend(
        [
            "",
            "## PGD White-Box Evaluation",
            "",
            "| Model | PGD robust accuracy |",
            "|---|---:|",
        ]
    )
    for model_key in MODEL_ORDER:
        lines.append(f"| {model_key} | {_format_percent(float(pgd_means.get(model_key, math.nan)))} |")

    defended_models = ["lenet_fgsm_at", "smallcnn_fgsm_at"]
    stronger_than_standard = [
        model_key
        for model_key in defended_models
        if robust_means.get(model_key, math.nan) > robust_means.get(
            model_key.replace("_fgsm_at", "_standard"), math.nan
        )
    ]
    lines.extend(
        [
            "",
            "## Main Observations",
            "",
            f"- FGSM robustness was compared at epsilon `{epsilon:g}` in `[0, 1]` pixel space.",
            f"- Defended models with higher FGSM robust accuracy than their standard counterpart: {stronger_than_standard}.",
            "- Transferability is directional: outgoing values summarize attacks generated by the row/source model, incoming values summarize attacks received by the target model.",
            "- PGD is reported only as white-box evaluation; no PGD transfer matrix is generated.",
            "",
            "## Interpretation Notes",
            "",
            "- All values are computed from CSV files produced by this run.",
            "- The 90% value is a reference target from 군 적용 참고 자료에서 제시한 작전운용성능 참고 목표치 90%, not a universal criterion.",
            "- Test data is used only for final evaluation.",
            "- NaN conditional transfer values indicate zero jointly clean-correct denominator.",
        ]
    )

    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_figures_index(figures_dir: str | Path, output_path: str | Path) -> None:
    """Write a small Markdown index for generated graph files."""
    figures_path = Path(figures_dir)
    output = Path(output_path)
    ensure_dir(output.parent)
    descriptions = {
        "robustness_curve.png": "FGSM epsilon별 robust accuracy 곡선",
        "clean_robust_comparison.png": "clean accuracy와 epsilon=0.25 FGSM robust accuracy 비교",
        "clean_accuracy_retention.png": "FGSM adversarial training 모델의 clean accuracy retention",
        "pgd_whitebox.png": "PGD L-infinity white-box robust accuracy 비교. 평균, 표준편차, seed별 점을 함께 표시",
        "adversarial_examples.png": "원본, FGSM, PGD 이미지 예시",
    }
    lines = ["# Figure Index", ""]
    for figure in sorted(figures_path.glob("*.png")):
        description = descriptions.get(figure.name, "FGSM 전이성 heatmap")
        lines.extend([f"## {figure.name}", "", description, "", f"![]({figure.name})", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
