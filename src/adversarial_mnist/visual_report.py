"""Build a visual storyboard from already generated experiment artifacts."""

from __future__ import annotations

import html
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib
import numpy as np
import pandas as pd
import torch

from adversarial_mnist.models import MODEL_ORDER
from adversarial_mnist.utils import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001


SMALLCNN_MODELS = ["smallcnn_standard", "smallcnn_fgsm_at"]
PRESENTATION_COLORS = {
    "clean": "#2f5d7c",
    "fgsm": "#d88c2d",
    "pgd": "#b64b4b",
    "firewall": "#2f7a55",
}


def _require_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Required visual report artifact is missing: {target}")
    return target


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def _format_percent_one_decimal(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NaN"
    return f"{value * 100:.1f}%"


def _load_torch_artifact(path: str | Path) -> dict[str, Any]:
    target = _require_file(path)
    try:
        loaded = torch.load(target, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(target, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a dictionary torch artifact at {target}.")
    return loaded


def _firewall_condition_means(firewall: pd.DataFrame) -> pd.DataFrame:
    """Return one mean row per Firewall model and condition across seeds."""
    required = {"model", "condition", "final_safe_accuracy", "reject_rate"}
    missing = required - set(firewall.columns)
    if missing:
        raise ValueError(f"Firewall results are missing required columns: {sorted(missing)}")
    return (
        firewall[firewall["model"].isin(SMALLCNN_MODELS)]
        .groupby(["model", "condition"], as_index=False)[["final_safe_accuracy", "reject_rate"]]
        .mean()
    )


def _bar_label(axis: plt.Axes, bars: Any, values: list[float]) -> None:
    for bar, value in zip(bars, values, strict=True):
        label = "NaN" if pd.isna(value) else f"{value * 100:.1f}%"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            0.0 if pd.isna(value) else value + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_training_progress(raw_dir: str | Path, output_path: str | Path) -> None:
    """Plot model training/validation progress from ``training_history.csv``."""
    source = _require_file(Path(raw_dir) / "training_history.csv")
    frame = pd.read_csv(source)
    required = {"model", "epoch", "train_accuracy", "validation_accuracy", "train_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(12, 7))
    axes = [plt.subplot(2, 1, 1), plt.subplot(2, 1, 2)]
    colors = {
        "lenet_standard": "#4c78a8",
        "smallcnn_standard": "#f58518",
        "lenet_fgsm_at": "#54a24b",
        "smallcnn_fgsm_at": "#b279a2",
    }
    for model_key in MODEL_ORDER:
        group = frame[frame["model"] == model_key]
        if group.empty:
            continue
        stats = group.groupby("epoch")[["train_accuracy", "validation_accuracy", "train_loss"]].mean()
        axes[0].plot(
            stats.index,
            stats["validation_accuracy"],
            marker="o",
            label=model_key,
            color=colors.get(model_key),
        )
        axes[1].plot(
            stats.index,
            stats["train_loss"],
            marker="o",
            label=model_key,
            color=colors.get(model_key),
        )
    axes[0].set_title("Model Training Progress: validation accuracy")
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2)
    axes[1].set_title("Model Training Progress: training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_autoencoder_progress(raw_dir: str | Path, output_path: str | Path) -> None:
    """Plot autoencoder reconstruction loss from ``autoencoder_history.csv``."""
    source = _require_file(Path(raw_dir) / "autoencoder_history.csv")
    frame = pd.read_csv(source)
    required = {"seed", "epoch", "train_reconstruction_loss", "validation_reconstruction_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    output = Path(output_path)
    ensure_dir(output.parent)
    plt.figure(figsize=(9, 5.5))
    for seed, group in frame.groupby("seed", sort=False):
        group = group.sort_values("epoch")
        plt.plot(
            group["epoch"],
            group["train_reconstruction_loss"],
            marker="o",
            label=f"seed {seed} train",
            color="#4c78a8",
        )
        plt.plot(
            group["epoch"],
            group["validation_reconstruction_loss"],
            marker="s",
            label=f"seed {seed} validation",
            color="#f58518",
        )
    plt.title("Autoencoder Purifier Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_pipeline_flow(output_path: str | Path) -> None:
    """Draw the project story from robustness evaluation to Firewall defense."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, axis = plt.subplots(figsize=(13, 4.8))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    steps = [
        ("1. Train CNNs", "LeNet / SmallCNN\nStandard / FGSM-AT", "#4c78a8"),
        ("2. Attack Evaluation", "Clean, FGSM curves\nTransfer matrix\nPGD white-box", "#f58518"),
        ("3. Diagnosis", "FGSM-AT looks strong\nPGD and transfer reveal limits", "#e45756"),
        ("4. Adversarial Firewall", "Detect reconstruction error\nPurify with autoencoder\nReject suspicious input", "#54a24b"),
        ("5. Visual Evidence", "CSV metrics\nFigures\nExample image grids", "#b279a2"),
    ]
    x_positions = np.linspace(0.11, 0.89, len(steps))
    for index, ((title, detail, color), x_position) in enumerate(zip(steps, x_positions, strict=True)):
        box = FancyBboxPatch(
            (x_position - 0.085, 0.36),
            0.17,
            0.34,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#f8f8f8",
        )
        axis.add_patch(box)
        axis.text(x_position, 0.62, title, ha="center", va="center", fontsize=11, weight="bold")
        axis.text(x_position, 0.48, detail, ha="center", va="center", fontsize=9)
        if index < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x_position + 0.095, 0.53),
                (x_positions[index + 1] - 0.095, 0.53),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.4,
                color="#444444",
            )
            axis.add_patch(arrow)
    axis.text(
        0.5,
        0.86,
        "MNIST Adversarial Robustness Study -> System-Level Defense Storyboard",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    axis.text(
        0.5,
        0.2,
        "The visual report uses existing checkpoints and result CSVs; it does not retrain models.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)


def plot_result_storyboard(raw_dir: str | Path, output_path: str | Path) -> None:
    """Create a single summary figure for the most important results."""
    raw_path = Path(raw_dir)
    clean = pd.read_csv(_require_file(raw_path / "clean_accuracy.csv"))
    fgsm = pd.read_csv(_require_file(raw_path / "fgsm_robustness.csv"))
    pgd = pd.read_csv(_require_file(raw_path / "pgd_whitebox.csv"))
    firewall = pd.read_csv(_require_file(raw_path / "firewall_results.csv"))

    clean_mean = clean.groupby("model")["clean_accuracy"].mean().reindex(MODEL_ORDER)
    fgsm_eps = fgsm[np.isclose(fgsm["epsilon"].astype(float), 0.25)]
    fgsm_mean = fgsm_eps.groupby("model")["robust_accuracy"].mean().reindex(MODEL_ORDER)
    pgd_mean = pgd.groupby("model")["robust_accuracy"].mean().reindex(MODEL_ORDER)

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    model_positions = np.arange(len(MODEL_ORDER))

    clean_values = [float(clean_mean.get(model, math.nan)) for model in MODEL_ORDER]
    fgsm_values = [float(fgsm_mean.get(model, math.nan)) for model in MODEL_ORDER]
    width = 0.38
    axes[0].bar(model_positions - width / 2, clean_values, width, label="Clean", color="#4c78a8")
    bars = axes[0].bar(
        model_positions + width / 2,
        fgsm_values,
        width,
        label="FGSM eps=0.25",
        color="#f58518",
    )
    _bar_label(axes[0], bars, fgsm_values)
    axes[0].set_title("Part 1: Clean vs FGSM Robust Accuracy")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_xticks(model_positions, MODEL_ORDER, rotation=25, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    pgd_values = [float(pgd_mean.get(model, math.nan)) for model in MODEL_ORDER]
    bars = axes[1].bar(MODEL_ORDER, pgd_values, color="#e45756")
    _bar_label(axes[1], bars, pgd_values)
    axes[1].set_title("Diagnosis: PGD White-Box Accuracy Drops")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(model_positions, MODEL_ORDER, rotation=25, ha="right")
    axes[1].set_ylabel("Robust accuracy")
    axes[1].grid(axis="y", alpha=0.3)

    firewall_subset = _firewall_condition_means(firewall)
    condition_order = ["Clean", "FGSM", "PGD"]
    x_positions = np.arange(len(condition_order))
    for offset, model_key, color in [
        (-0.18, "smallcnn_standard", "#4c78a8"),
        (0.18, "smallcnn_fgsm_at", "#54a24b"),
    ]:
        rows = firewall_subset[firewall_subset["model"] == model_key].set_index("condition")
        values = [float(rows.loc[condition, "final_safe_accuracy"]) for condition in condition_order]
        bars = axes[2].bar(x_positions + offset, values, 0.34, label=model_key, color=color)
        _bar_label(axes[2], bars, values)
    axes[2].set_title("Part 2: Firewall Final Safe Accuracy")
    axes[2].set_ylim(0.0, 1.08)
    axes[2].set_xticks(x_positions, condition_order)
    axes[2].set_ylabel("Final safe accuracy")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(loc="lower left")

    for offset, model_key, color in [
        (-0.18, "smallcnn_standard", "#b279a2"),
        (0.18, "smallcnn_fgsm_at", "#72b7b2"),
    ]:
        rows = firewall_subset[firewall_subset["model"] == model_key].set_index("condition")
        values = [float(rows.loc[condition, "reject_rate"]) for condition in condition_order]
        bars = axes[3].bar(x_positions + offset, values, 0.34, label=model_key, color=color)
        _bar_label(axes[3], bars, values)
    axes[3].set_title("Firewall Reject Option Usage")
    axes[3].set_ylim(0.0, 0.35)
    axes[3].set_xticks(x_positions, condition_order)
    axes[3].set_ylabel("Reject rate")
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].legend()

    fig.suptitle("Visual Result Storyboard", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)


def plot_attack_defense_ladder(raw_dir: str | Path, output_path: str | Path) -> None:
    """Plot the central story: FGSM looks strong, PGD breaks it, Firewall recovers."""
    raw_path = Path(raw_dir)
    clean = pd.read_csv(_require_file(raw_path / "clean_accuracy.csv"))
    fgsm = pd.read_csv(_require_file(raw_path / "fgsm_robustness.csv"))
    pgd10 = pd.read_csv(_require_file(raw_path / "pgd_whitebox.csv"))
    pgd20 = pd.read_csv(_require_file(raw_path / "pgd20_restart5_whitebox.csv"))
    firewall = pd.read_csv(_require_file(raw_path / "firewall_results.csv"))

    model_key = "smallcnn_fgsm_at"
    clean_value = float(clean[clean["model"] == model_key]["clean_accuracy"].mean())
    fgsm_value = float(
        fgsm[(fgsm["model"] == model_key) & np.isclose(fgsm["epsilon"].astype(float), 0.25)][
            "robust_accuracy"
        ].mean()
    )
    pgd10_value = float(pgd10[pgd10["model"] == model_key]["robust_accuracy"].mean())
    pgd20_value = float(pgd20[pgd20["model"] == model_key]["robust_accuracy"].mean())
    firewall_value = float(
        firewall[(firewall["model"] == model_key) & (firewall["condition"] == "PGD")][
            "final_safe_accuracy"
        ].mean()
    )

    stages = [
        ("Clean\naccuracy", clean_value, PRESENTATION_COLORS["clean"], "Normal performance is preserved"),
        ("FGSM\nrobust", fgsm_value, PRESENTATION_COLORS["fgsm"], "Single-step attack looks defended"),
        ("PGD-10\nrobust", pgd10_value, PRESENTATION_COLORS["pgd"], "Iterative attack exposes collapse"),
        ("PGD-20\n5 restarts", pgd20_value, "#7f4c8a", "Stronger PGD confirms the weakness"),
        ("Firewall PGD\nfinal safe", firewall_value, PRESENTATION_COLORS["firewall"], "Input-stage defense recovers safety"),
    ]
    output = Path(output_path)
    ensure_dir(output.parent)
    fig, axis = plt.subplots(figsize=(14, 6.8))
    axis.set_facecolor("#fbfcfe")
    x_positions = np.arange(len(stages))
    values = [stage[1] for stage in stages]
    colors = [stage[2] for stage in stages]

    axis.plot(x_positions, values, color="#b8c1cc", linewidth=3, zorder=1)
    axis.scatter(x_positions, values, s=520, c=colors, edgecolors="white", linewidths=3, zorder=3)
    for index, (_label, value, color, note) in enumerate(stages):
        axis.text(
            index,
            value + 0.065 if value < 0.9 else value - 0.11,
            _format_percent_one_decimal(value),
            ha="center",
            va="bottom" if value < 0.9 else "top",
            fontsize=18,
            weight="bold",
            color=color,
        )
        axis.text(
            index,
            -0.13,
            note,
            ha="center",
            va="top",
            fontsize=10,
            color="#4a535d",
            wrap=True,
        )
    axis.axhspan(0.9, 1.0, color="#dceee5", alpha=0.7, label="90% reference zone")
    axis.set_xticks(x_positions, [stage[0] for stage in stages], fontsize=12)
    axis.set_ylim(-0.22, 1.08)
    axis.set_ylabel("Accuracy / final safe accuracy")
    axis.set_title(
        "SmallCNN FGSM-AT Story: strong FGSM score, PGD collapse, Firewall recovery",
        fontsize=16,
        weight="bold",
        pad=18,
    )
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", frameon=False)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close(fig)


def plot_presentation_dashboard(raw_dir: str | Path, output_path: str | Path) -> None:
    """Create a single 16:9 presentation dashboard for the whole project."""
    raw_path = Path(raw_dir)
    clean = pd.read_csv(_require_file(raw_path / "clean_accuracy.csv"))
    fgsm = pd.read_csv(_require_file(raw_path / "fgsm_robustness.csv"))
    pgd10 = pd.read_csv(_require_file(raw_path / "pgd_whitebox.csv"))
    pgd20 = pd.read_csv(_require_file(raw_path / "pgd20_restart5_whitebox.csv"))
    firewall = pd.read_csv(_require_file(raw_path / "firewall_results.csv"))

    model_key = "smallcnn_fgsm_at"
    clean_value = float(clean[clean["model"] == model_key]["clean_accuracy"].mean())
    fgsm_value = float(
        fgsm[(fgsm["model"] == model_key) & np.isclose(fgsm["epsilon"].astype(float), 0.25)][
            "robust_accuracy"
        ].mean()
    )
    pgd10_value = float(pgd10[pgd10["model"] == model_key]["robust_accuracy"].mean())
    pgd20_value = float(pgd20[pgd20["model"] == model_key]["robust_accuracy"].mean())
    firewall_pgd = float(
        firewall[(firewall["model"] == model_key) & (firewall["condition"] == "PGD")][
            "final_safe_accuracy"
        ].mean()
    )
    firewall_standard_pgd = float(
        firewall[(firewall["model"] == "smallcnn_standard") & (firewall["condition"] == "PGD")][
            "final_safe_accuracy"
        ].mean()
    )

    output = Path(output_path)
    ensure_dir(output.parent)
    fig = plt.figure(figsize=(16, 9), facecolor="#f4f6f8")
    grid = fig.add_gridspec(3, 4, height_ratios=[0.45, 1.1, 1.25], hspace=0.42, wspace=0.32)

    title_axis = fig.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.72,
        "FGSM adversarial training is not enough: PGD and transfer reveal limits",
        fontsize=23,
        weight="bold",
        color="#17202c",
        transform=title_axis.transAxes,
    )
    title_axis.text(
        0.0,
        0.20,
        "3 seeds, full MNIST test set. Firewall values are final safe accuracy, not ordinary robust accuracy.",
        fontsize=12,
        color="#5c6670",
        transform=title_axis.transAxes,
    )

    cards = [
        ("Clean kept", clean_value, "SmallCNN FGSM-AT", PRESENTATION_COLORS["clean"]),
        ("FGSM robust", fgsm_value, "same attack, eps=0.25", PRESENTATION_COLORS["fgsm"]),
        ("PGD-20 robust", pgd20_value, "mean, 3 seeds, 5 restarts", PRESENTATION_COLORS["pgd"]),
        ("Firewall PGD safe", firewall_pgd, "detect + purify + reject", PRESENTATION_COLORS["firewall"]),
    ]
    for index, (title, value, detail, color) in enumerate(cards):
        axis = fig.add_subplot(grid[1, index])
        axis.set_facecolor("white")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d8dee7")
        axis.text(0.06, 0.78, title, fontsize=13, weight="bold", color="#27313c", transform=axis.transAxes)
        axis.text(0.06, 0.36, _format_percent_one_decimal(value), fontsize=30, weight="bold", color=color, transform=axis.transAxes)
        axis.text(0.06, 0.14, detail, fontsize=10.5, color="#5c6670", transform=axis.transAxes)
        axis.axhline(0.04, xmin=0.06, xmax=0.94, color=color, linewidth=4)

    axis = fig.add_subplot(grid[2, :2])
    ladder_labels = ["Clean", "FGSM", "PGD-10", "PGD-20", "Firewall"]
    ladder_values = [clean_value, fgsm_value, pgd10_value, pgd20_value, firewall_pgd]
    ladder_colors = [
        PRESENTATION_COLORS["clean"],
        PRESENTATION_COLORS["fgsm"],
        PRESENTATION_COLORS["pgd"],
        "#7f4c8a",
        PRESENTATION_COLORS["firewall"],
    ]
    x_positions = np.arange(len(ladder_labels))
    axis.plot(x_positions, ladder_values, color="#bdc7d2", linewidth=3)
    axis.scatter(x_positions, ladder_values, s=250, color=ladder_colors, edgecolor="white", linewidth=2, zorder=3)
    for x_position, value, color in zip(x_positions, ladder_values, ladder_colors, strict=True):
        axis.text(x_position, value + 0.05, _format_percent_one_decimal(value), ha="center", color=color, weight="bold")
    axis.set_xticks(x_positions, ladder_labels)
    axis.set_ylim(0, 1.08)
    axis.set_title("SmallCNN FGSM-AT trajectory")
    axis.set_ylabel("Accuracy / final safe accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)

    axis = fig.add_subplot(grid[2, 2:])
    condition_order = ["Clean", "FGSM", "PGD"]
    means = (
        firewall.groupby(["model", "condition"])["final_safe_accuracy"]
        .mean()
        .unstack()
        .reindex(index=["smallcnn_standard", "smallcnn_fgsm_at"], columns=condition_order)
    )
    bar_width = 0.36
    positions = np.arange(len(condition_order))
    for offset, row_key, color, label in [
        (-bar_width / 2, "smallcnn_standard", "#4e79a7", "SmallCNN Standard"),
        (bar_width / 2, "smallcnn_fgsm_at", PRESENTATION_COLORS["firewall"], "SmallCNN FGSM-AT"),
    ]:
        values = means.loc[row_key].to_numpy(dtype=float)
        bars = axis.bar(positions + offset, values, bar_width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                _format_percent_one_decimal(float(value)),
                ha="center",
                fontsize=9,
                color=color,
                weight="bold",
            )
    axis.axhline(0.9, color="#555555", linestyle="--", linewidth=1, label="90% reference")
    axis.set_xticks(positions, condition_order)
    axis.set_ylim(0, 1.08)
    axis.set_title("Firewall final safe accuracy")
    axis.set_ylabel("Final safe accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.99,
        0.01,
        f"Key PGD recovery: SmallCNN standard {firewall_standard_pgd * 100:.2f}%, SmallCNN FGSM-AT {firewall_pgd * 100:.2f}%",
        ha="right",
        fontsize=10,
        color="#5c6670",
    )
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_firewall_pipeline_strip(raw_dir: str | Path, output_path: str | Path) -> None:
    """Create a readable sample-level Firewall pipeline strip from saved examples."""
    raw_path = Path(raw_dir)
    examples = _load_torch_artifact(raw_path / "firewall_examples.pt")
    condition = "PGD"
    condition_examples = examples.get("conditions", {}).get(condition)
    if not isinstance(condition_examples, dict):
        raise ValueError("firewall_examples.pt does not contain PGD examples.")

    labels = condition_examples["labels"]
    columns = min(4, int(labels.shape[0]))
    output = Path(output_path)
    ensure_dir(output.parent)
    fig, axes = plt.subplots(3, columns, figsize=(columns * 3.0, 7.2), facecolor="#f4f6f8")
    if columns == 1:
        axes = np.asarray(axes).reshape(3, 1)
    row_specs = [
        ("Original", "original", PRESENTATION_COLORS["clean"]),
        ("PGD input", "input", PRESENTATION_COLORS["pgd"]),
        ("Purified", "purified", PRESENTATION_COLORS["firewall"]),
    ]
    for row_index, (row_label, tensor_key, color) in enumerate(row_specs):
        for column_index in range(columns):
            axis = axes[row_index, column_index]
            axis.imshow(
                condition_examples[tensor_key][column_index, 0].detach().numpy(),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            if column_index == 0:
                axis.set_ylabel(row_label, fontsize=12, weight="bold", color=color)
            if row_index == 0:
                axis.set_title(f"label {int(labels[column_index])}", fontsize=11)
            if row_index == 1:
                pred = int(condition_examples["input_predictions"][column_index])
                axis.text(0.5, -0.08, f"input pred {pred}", transform=axis.transAxes, ha="center", va="top", fontsize=9)
            if row_index == 2:
                pred = int(condition_examples["purified_predictions"][column_index])
                score = float(condition_examples["scores"][column_index])
                decision = str(condition_examples["decisions"][column_index]).replace("_", " ")
                axis.text(
                    0.5,
                    -0.08,
                    f"purified pred {pred}\nscore {score:.4f}\n{decision}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
    fig.suptitle(
        "Adversarial Firewall sample flow: original -> PGD input -> purified decision",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Examples are embedded from results/raw/firewall_examples.pt; the figure replays saved outputs and does not rerun attacks.",
        ha="center",
        fontsize=10,
        color="#5c6670",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.93))
    plt.savefig(output, dpi=220)
    plt.close(fig)


def _checkpoint_inventory(checkpoints_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(checkpoints_dir).glob("*.pt")):
        rows.append(
            {
                "file": path.name,
                "size_mb": path.stat().st_size / (1024 * 1024),
                "modified": path.stat().st_mtime,
            }
        )
    return pd.DataFrame(rows)


def _metrics_for_html(config: dict[str, Any]) -> dict[str, str]:
    raw_dir = Path(config["paths"]["raw_dir"])
    clean = pd.read_csv(_require_file(raw_dir / "clean_accuracy.csv"))
    fgsm = pd.read_csv(_require_file(raw_dir / "fgsm_robustness.csv"))
    pgd = pd.read_csv(_require_file(raw_dir / "pgd_whitebox.csv"))
    pgd20 = pd.read_csv(_require_file(raw_dir / "pgd20_restart5_whitebox.csv"))
    firewall = pd.read_csv(_require_file(raw_dir / "firewall_results.csv"))

    clean_mean = clean.groupby("model")["clean_accuracy"].mean()
    fgsm_025 = fgsm[np.isclose(fgsm["epsilon"].astype(float), 0.25)]
    fgsm_mean = fgsm_025.groupby("model")["robust_accuracy"].mean()
    pgd_mean = pgd.groupby("model")["robust_accuracy"].mean()
    pgd20_mean = pgd20.groupby("model")["robust_accuracy"].mean()
    firewall_pgd = firewall[
        (firewall["model"] == "smallcnn_fgsm_at") & (firewall["condition"] == "PGD")
    ]
    firewall_pgd_safe = float(firewall_pgd["final_safe_accuracy"].mean())
    if firewall_pgd.empty:
        firewall_pgd_safe = math.nan
    return {
        "smallcnn_clean": _format_percent(float(clean_mean.get("smallcnn_fgsm_at", math.nan))),
        "smallcnn_fgsm": _format_percent(float(fgsm_mean.get("smallcnn_fgsm_at", math.nan))),
        "smallcnn_pgd": _format_percent(float(pgd_mean.get("smallcnn_fgsm_at", math.nan))),
        "smallcnn_pgd20": _format_percent(float(pgd20_mean.get("smallcnn_fgsm_at", math.nan))),
        "firewall_pgd_safe": _format_percent(firewall_pgd_safe),
        "firewall_seed_note": "Firewall 값은 seed 42, 123, 2026 full test 평균입니다.",
    }


def _figure_card(title: str, image_path: str, caption: str) -> str:
    return f"""
        <article class="figure-card">
          <h3>{html.escape(title)}</h3>
          <img src="{html.escape(image_path)}" alt="{html.escape(title)}">
          <p>{html.escape(caption)}</p>
        </article>
    """


def _checkpoint_table(checkpoints: pd.DataFrame) -> str:
    if checkpoints.empty:
        return "<p>No checkpoint files found.</p>"
    rows = []
    for _, row in checkpoints.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['file']))}</td>"
            f"<td>{float(row['size_mb']):.2f} MB</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Checkpoint</th><th>Size</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def build_visual_report(config: dict[str, Any]) -> Path:
    """Generate figures and a static HTML visual report from existing outputs."""
    from adversarial_mnist.visualization import generate_figures_index

    results_dir = Path(config["paths"]["results_dir"])
    raw_dir = Path(config["paths"]["raw_dir"])
    figures_dir = ensure_dir(config["paths"]["figures_dir"])
    report_dir = ensure_dir(results_dir / "visual_report")

    plot_training_progress(raw_dir, figures_dir / "training_progress.png")
    plot_autoencoder_progress(raw_dir, figures_dir / "autoencoder_training_progress.png")
    plot_pipeline_flow(figures_dir / "project_pipeline_flow.png")
    plot_result_storyboard(raw_dir, figures_dir / "result_storyboard.png")
    plot_presentation_dashboard(raw_dir, figures_dir / "presentation_dashboard.png")
    plot_attack_defense_ladder(raw_dir, figures_dir / "attack_defense_ladder.png")
    plot_firewall_pipeline_strip(raw_dir, figures_dir / "firewall_pipeline_strip.png")
    generate_figures_index(figures_dir, figures_dir / "figure_index.md")

    metrics = _metrics_for_html(config)
    checkpoints = _checkpoint_inventory(config["paths"]["checkpoints_dir"])
    checkpoint_count = int(checkpoints.shape[0])

    cards = [
        _figure_card(
            "Attack-Defense Ladder",
            "../figures/attack_defense_ladder.png",
            "A presentation-friendly line showing clean performance, FGSM robustness, PGD collapse, and Firewall recovery.",
        ),
        _figure_card(
            "Firewall Pipeline Strip",
            "../figures/firewall_pipeline_strip.png",
            "Saved PGD examples replayed as original, attacked input, and purified Firewall decision.",
        ),
        _figure_card(
            "Project Flow",
            "../figures/project_pipeline_flow.png",
            "The whole story: train models, attack them, diagnose limits, then add Firewall defense.",
        ),
        _figure_card(
            "Training Progress",
            "../figures/training_progress.png",
            "Validation accuracy and training loss from the previously trained CNN checkpoints.",
        ),
        _figure_card(
            "Core Result Storyboard",
            "../figures/result_storyboard.png",
            "One slide that connects FGSM robustness, PGD collapse, and Firewall final safe accuracy.",
        ),
        _figure_card(
            "Adversarial Examples",
            "../figures/adversarial_examples.png",
            "Original, FGSM, and PGD examples used in the robustness study.",
        ),
        _figure_card(
            "Firewall Image Flow",
            "../figures/original_attacked_purified_examples.png",
            "Original, attacked/input, and autoencoder-purified images with predictions and decisions.",
        ),
        _figure_card(
            "Firewall Detection Scores",
            "../figures/firewall_score_distribution.png",
            "Reconstruction-error distributions for clean, FGSM, and PGD inputs.",
        ),
        _figure_card(
            "Firewall Decisions",
            "../figures/firewall_decision_breakdown.png",
            "How often the system accepted original inputs, accepted purified inputs, or rejected suspicious inputs.",
        ),
        _figure_card(
            "Strong PGD Check",
            "../figures/pgd20_restart5_whitebox.png",
            "PGD-20 with five restarts shows seed instability and limits of FGSM-only robustness.",
        ),
    ]

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adversarial MNIST Visual Report</title>
  <style>
    :root {{
      --ink: #1e2329;
      --muted: #5b6470;
      --line: #d9dee7;
      --panel: #ffffff;
      --bg: #f4f6f8;
      --blue: #315f92;
      --green: #3d7c5f;
      --red: #a44545;
      --amber: #9b6b21;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Malgun Gothic", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      padding: 28px 34px 18px;
      background: #111827;
      color: white;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    header p {{
      max-width: 980px;
      margin: 0;
      color: #d1d5db;
      font-size: 16px;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    section {{
      margin: 0 0 22px;
      padding: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      padding: 14px;
      border-radius: 6px;
      background: #fbfcfe;
    }}
    .metric strong {{ display: block; font-size: 28px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .hero-figure {{
      margin: 18px 0 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
    }}
    .hero-figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
    }}
    .figure-card h3 {{ margin: 0 0 10px; font-size: 18px; }}
    .figure-card img {{
      width: 100%;
      height: auto;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      background: white;
    }}
    .figure-card p {{ margin: 9px 0 0; color: var(--muted); font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: #f0f3f7; }}
    .note {{ color: var(--muted); }}
    .links a {{
      display: inline-block;
      margin: 4px 10px 4px 0;
      color: var(--blue);
      text-decoration: none;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Adversarial MNIST Visual Report</h1>
    <p>이미 학습된 checkpoint와 저장된 CSV/PNG를 사용해, FGSM 방어의 한계와 Adversarial Firewall 확장을 발표용 흐름으로 재구성한 정적 보고서입니다.</p>
  </header>
  <main>
    <section>
      <h2>핵심 숫자</h2>
      <div class="metrics">
        <div class="metric"><span>SmallCNN FGSM-AT clean accuracy</span><strong>{metrics["smallcnn_clean"]}</strong></div>
        <div class="metric"><span>SmallCNN FGSM-AT FGSM eps=0.25</span><strong>{metrics["smallcnn_fgsm"]}</strong></div>
        <div class="metric"><span>SmallCNN FGSM-AT PGD-20 restart 5</span><strong>{metrics["smallcnn_pgd20"]}</strong></div>
        <div class="metric"><span>Firewall PGD final safe accuracy</span><strong>{metrics["firewall_pgd_safe"]}</strong></div>
      </div>
      <p class="note">{metrics["firewall_seed_note"]} 기존 FGSM/PGD/전이성 실험도 seed 42, 123, 2026 결과를 포함합니다.</p>
      <div class="hero-figure">
        <img src="../figures/presentation_dashboard.png" alt="Presentation dashboard">
      </div>
    </section>

    <section>
      <h2>시각 스토리보드</h2>
      <div class="grid">
        {"".join(cards)}
      </div>
    </section>

    <section>
      <h2>사용한 학습 산출물</h2>
      <p>현재 checkpoint 파일 {checkpoint_count}개를 기준으로 보고서를 생성했습니다.</p>
      {_checkpoint_table(checkpoints)}
    </section>

    <section>
      <h2>관련 파일</h2>
      <div class="links">
        <a href="../simulation/index.html">interactive simulation</a>
        <a href="../summary.md">results/summary.md</a>
        <a href="../../업데이트.md">업데이트.md</a>
        <a href="../figures/figure_index.md">figure_index.md</a>
        <a href="../raw/firewall_results.csv">firewall_results.csv</a>
        <a href="../raw/fgsm_robustness.csv">fgsm_robustness.csv</a>
      </div>
    </section>
  </main>
</body>
</html>
"""
    output = report_dir / "index.html"
    output.write_text(html_text, encoding="utf-8")
    return output
