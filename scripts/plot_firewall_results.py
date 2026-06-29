"""Generate Adversarial Firewall figures and update summary.md."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.utils import setup_logging
from adversarial_mnist.visualization import (
    plot_firewall_results,
    update_firewall_summary_markdown,
)
from scripts.common import add_common_args, prepare_config

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser


def main() -> None:
    setup_logging()
    args = build_parser().parse_args()
    config = prepare_config(args)
    plot_firewall_results(config)
    update_firewall_summary_markdown(config, f"{config['paths']['results_dir']}/summary.md")
    LOGGER.info("Generated Firewall figures and updated summary.md.")


if __name__ == "__main__":
    main()
