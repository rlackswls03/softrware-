"""Evaluate PGD L-infinity white-box robustness for trained models."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import (
    evaluate_all_pgd_whitebox,
    load_trained_models,
    save_adversarial_example_batch,
    write_csv,
)
from adversarial_mnist.utils import set_reproducibility, setup_logging
from scripts.common import add_common_args, prepare_config, prepare_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    device = prepare_device(args)
    rows: list[dict[str, object]] = []
    first_models = None
    first_loader = None
    for seed in config["seeds"]:
        set_reproducibility(int(seed), deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=int(seed))
        models = load_trained_models(
            config["models"],
            config["paths"]["checkpoints_dir"],
            seed=int(seed),
            device=device,
        )
        if first_models is None:
            first_models = models
            first_loader = loaders.test
        rows.extend(
            evaluate_all_pgd_whitebox(
                models,
                loaders.test,
                int(seed),
                device,
                config["evaluation"]["pgd"],
            )
        )
    write_csv(rows, f"{config['paths']['raw_dir']}/pgd_whitebox.csv")
    if first_models is not None and first_loader is not None:
        save_adversarial_example_batch(
            first_models[config["models"][0]],
            first_loader,
            f"{config['paths']['raw_dir']}/adversarial_examples.pt",
            device=device,
            epsilon=float(config["evaluation"]["pgd"]["epsilon"]),
            pgd_steps=int(config["evaluation"]["pgd"]["steps"]),
        )
    LOGGER.info("Wrote PGD white-box CSV and adversarial example tensor batch.")


if __name__ == "__main__":
    main()
