"""Build a static visual report from existing checkpoints and result files."""

from __future__ import annotations

import argparse
import logging

from adversarial_mnist.utils import setup_logging
from adversarial_mnist.visual_report import build_visual_report
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
    output = build_visual_report(config)
    LOGGER.info("Visual report written to %s.", output)


if __name__ == "__main__":
    main()
