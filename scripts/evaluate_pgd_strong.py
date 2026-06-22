"""Evaluate PGD-20 with five random restarts on a fixed test subset."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import pandas as pd

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import (
    evaluate_pgd_whitebox_restarts,
    load_trained_models,
    write_csv,
)
from adversarial_mnist.metrics import aggregate_mean_std
from adversarial_mnist.utils import ensure_dir, set_reproducibility, setup_logging
from adversarial_mnist.visualization import plot_pgd_whitebox_bar
from scripts.common import add_common_args, prepare_config, prepare_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--test-subset", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--test-batch-size", type=int, default=128)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    device = prepare_device(args)
    eval_config = copy.deepcopy(config)
    eval_config["dataset"]["test_subset"] = args.test_subset
    eval_config["training"]["test_batch_size"] = args.test_batch_size

    output_csv = Path(config["paths"]["raw_dir"]) / "pgd20_restart5_whitebox.csv"
    if output_csv.exists() and not args.force:
        raise FileExistsError(f"{output_csv} already exists. Use --force to overwrite.")

    rows: list[dict[str, object]] = []
    for seed in config["seeds"]:
        seed = int(seed)
        set_reproducibility(seed, deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(eval_config, seed=seed)
        models = load_trained_models(
            config["models"],
            config["paths"]["checkpoints_dir"],
            seed=seed,
            device=device,
        )
        for model_key, model in models.items():
            LOGGER.info(
                "Evaluating %s seed=%s with PGD-%s restarts=%s on %s samples.",
                model_key,
                seed,
                args.steps,
                args.restarts,
                loaders.test_size,
            )
            row = evaluate_pgd_whitebox_restarts(
                model_key,
                model,
                loaders.test,
                seed,
                device,
                epsilon=args.epsilon,
                steps=args.steps,
                restarts=args.restarts,
                alpha=args.alpha,
            )
            row["test_subset"] = loaders.test_size
            rows.append(row)
            LOGGER.info(
                "%s seed=%s robust_accuracy=%.4f",
                model_key,
                seed,
                row["robust_accuracy"],
            )

    write_csv(rows, output_csv)
    frame = pd.DataFrame(rows)
    summary = aggregate_mean_std(
        frame,
        group_columns=["model", "architecture", "training", "attack", "epsilon", "pgd_steps", "pgd_restarts"],
        metric_columns=["robust_accuracy", "attack_success_rate"],
    )
    aggregated_dir = ensure_dir(config["paths"]["aggregated_dir"])
    summary.to_csv(aggregated_dir / "pgd20_restart5_summary.csv", index=False)

    figures_dir = ensure_dir(config["paths"]["figures_dir"])
    plot_pgd_whitebox_bar(output_csv, figures_dir / "pgd20_restart5_whitebox.png")
    LOGGER.info("Wrote %s and PGD-20 restart-5 figure.", output_csv)


if __name__ == "__main__":
    main()
