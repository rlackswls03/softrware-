"""Shared helpers for command-line scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from adversarial_mnist.utils import (
    apply_quick_overrides,
    choose_device,
    ensure_dir,
    load_json,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add config, quick, seeds, and device CLI arguments."""
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--quick", action="store_true", help="Run reduced connectivity check mode.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite generated outputs.")


def prepare_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load config and apply CLI overrides."""
    config = load_json(args.config)
    if args.quick:
        config = apply_quick_overrides(config)
    if args.seeds is not None:
        config["seeds"] = args.seeds
    return config


def prepare_device(args: argparse.Namespace) -> torch.device:
    """Select a torch device from CLI arguments."""
    return choose_device(args.device)


def save_run_config(config: dict[str, Any]) -> None:
    """Persist the effective run config."""
    results_dir = ensure_dir(config["paths"]["results_dir"])
    save_json(config, results_dir / "run_config.json")


def remove_generated_results(config: dict[str, Any]) -> None:
    """Delete generated result files under the configured results directory."""
    results_dir = Path(config["paths"]["results_dir"])
    if not results_dir.exists():
        return
    patterns = [
        "*.json",
        "*.md",
        "raw/*.csv",
        "raw/*.pt",
        "aggregated/*.csv",
        "figures/*.png",
    ]
    for pattern in patterns:
        for path in results_dir.glob(pattern):
            if path.is_file():
                LOGGER.info("Removing generated result file %s", path)
                path.unlink()
