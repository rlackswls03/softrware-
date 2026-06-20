"""Evaluate clean and FGSM robustness for trained models."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import (
    evaluate_all_clean,
    evaluate_all_fgsm_robustness,
    load_trained_models,
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
    clean_rows: list[dict[str, object]] = []
    fgsm_rows: list[dict[str, object]] = []
    for seed in config["seeds"]:
        set_reproducibility(int(seed), deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=int(seed))
        models = load_trained_models(
            config["models"],
            config["paths"]["checkpoints_dir"],
            seed=int(seed),
            device=device,
        )
        clean_rows.extend(evaluate_all_clean(models, loaders.test, int(seed), device))
        fgsm_rows.extend(
            evaluate_all_fgsm_robustness(
                models,
                loaders.test,
                config["evaluation"]["fgsm_epsilons"],
                int(seed),
                device,
            )
        )
    write_csv(clean_rows, f"{config['paths']['raw_dir']}/clean_accuracy.csv")
    write_csv(fgsm_rows, f"{config['paths']['raw_dir']}/fgsm_robustness.csv")
    LOGGER.info("Wrote clean and FGSM robustness CSV files.")


if __name__ == "__main__":
    main()
