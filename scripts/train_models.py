"""Train the four configured MNIST models."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.training import train_model
from adversarial_mnist.utils import append_csv_rows, set_reproducibility, setup_logging
from scripts.common import add_common_args, prepare_config, prepare_device

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    device = prepare_device(args)
    rows: list[dict[str, object]] = []
    for seed in config["seeds"]:
        set_reproducibility(int(seed), deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=int(seed))
        for model_key in config["models"]:
            _, model_rows = train_model(
                model_key,
                config,
                seed=int(seed),
                train_loader=loaders.train,
                validation_loader=loaders.validation,
                device=device,
                force=args.force,
                resume=args.resume,
                progress=not args.no_progress,
            )
            rows.extend(model_rows)
    if rows:
        append_csv_rows(rows, f"{config['paths']['raw_dir']}/training_history.csv")
        LOGGER.info("Wrote %s training history rows.", len(rows))
    else:
        LOGGER.info("No new training rows were produced.")


if __name__ == "__main__":
    main()
