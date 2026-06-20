"""Generate figures, aggregate CSVs, and summary Markdown from result CSV files."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.evaluation import aggregate_result_files
from adversarial_mnist.utils import setup_logging
from adversarial_mnist.visualization import generate_summary_markdown, plot_all_results
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
    aggregate_result_files(config["paths"]["raw_dir"], config["paths"]["aggregated_dir"])
    plot_all_results(config)
    generate_summary_markdown(config, f"{config['paths']['results_dir']}/summary.md")
    LOGGER.info("Generated figures and summary.md.")


if __name__ == "__main__":
    main()
