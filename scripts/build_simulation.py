"""Build an interactive Adversarial Firewall simulation from saved artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from adversarial_mnist.simulation import build_simulation
from adversarial_mnist.utils import setup_logging
from scripts.common import add_common_args, prepare_config

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=None,
        help="Path to results/raw/firewall_examples.pt.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Path to results/raw/firewall_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where index.html will be written.",
    )
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    output = build_simulation(
        config,
        examples_path=args.examples_path,
        metrics_path=args.metrics_path,
        output_dir=args.output_dir,
    )
    LOGGER.info("Interactive simulation written to %s.", output)


if __name__ == "__main__":
    main()
