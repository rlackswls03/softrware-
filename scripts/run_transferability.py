"""Run 4x4 FGSM transferability evaluation."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import evaluate_transferability, load_trained_models, write_csv
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
    order = config["evaluation"]["transfer_model_order"]
    for seed in config["seeds"]:
        set_reproducibility(int(seed), deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=int(seed))
        models = load_trained_models(order, config["paths"]["checkpoints_dir"], int(seed), device)
        rows.extend(
            evaluate_transferability(
                models,
                loaders.test,
                config["evaluation"]["fgsm_epsilons"],
                int(seed),
                device,
                model_order=order,
            )
        )
    write_csv(rows, f"{config['paths']['raw_dir']}/transferability_long.csv")
    LOGGER.info("Wrote transferability CSV. Rows are source models, columns are target models.")


if __name__ == "__main__":
    main()
