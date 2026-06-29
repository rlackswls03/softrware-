"""Run the Adversarial Firewall MVP pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from adversarial_mnist.training import checkpoint_path
from adversarial_mnist.utils import (
    save_json,
    set_reproducibility,
    setup_logging,
    write_environment,
)
from adversarial_mnist.visualization import (
    plot_firewall_results,
    update_firewall_summary_markdown,
)
from scripts.common import add_common_args, prepare_config, prepare_device
from scripts.evaluate_firewall import evaluate_firewall
from scripts.train_autoencoder import train_autoencoders

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--autoencoder-epochs", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--pgd-steps", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--threshold-fpr", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def _selected_models(config: dict[str, Any], cli_models: list[str] | None) -> list[str]:
    if cli_models:
        return cli_models
    firewall = config.get("firewall", {})
    if isinstance(firewall, dict):
        models = firewall.get("models", ["smallcnn_standard", "smallcnn_fgsm_at"])
        if isinstance(models, list):
            return [str(model) for model in models]
    return ["smallcnn_standard", "smallcnn_fgsm_at"]


def _check_classifier_checkpoints(config: dict[str, Any], model_keys: list[str]) -> None:
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    missing: list[Path] = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for model_key in model_keys:
            path = checkpoint_path(checkpoints_dir, model_key, seed, "last")
            if not path.exists():
                missing.append(path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing classifier checkpoints required for Firewall evaluation:\n"
            f"{formatted}\n"
            "Run `python -m scripts.train_models --config configs/default.json` first."
        )


def run() -> None:
    args = build_parser().parse_args()
    config = prepare_config(args)
    if args.test_subset is not None:
        config["dataset"]["test_subset"] = args.test_subset
    if args.autoencoder_epochs is not None:
        config.setdefault("firewall", {}).setdefault("autoencoder", {})[
            "epochs"
        ] = args.autoencoder_epochs
    device = prepare_device(args)
    model_keys = _selected_models(config, args.models)

    write_environment(f"{config['paths']['results_dir']}/firewall_environment.json", device=device)
    save_json(config, f"{config['paths']['results_dir']}/firewall_run_config.json")
    if config.get("quick", {}).get("enabled"):
        LOGGER.warning(
            "Firewall quick mode is only for connectivity checks; do not draw performance conclusions."
        )

    _check_classifier_checkpoints(config, model_keys)
    for seed_value in config["seeds"]:
        set_reproducibility(int(seed_value), deterministic=bool(config["training"]["deterministic"]))

    autoencoder_history = train_autoencoders(
        config,
        device,
        force=args.force,
        progress=not args.no_progress,
    )
    if autoencoder_history:
        from adversarial_mnist.evaluation import write_csv

        write_csv(autoencoder_history, f"{config['paths']['raw_dir']}/autoencoder_history.csv")

    evaluate_firewall(
        config,
        device,
        model_keys,
        force=args.force,
        epsilon=args.epsilon,
        pgd_steps=args.pgd_steps,
        min_confidence=args.min_confidence,
        threshold_fpr=args.threshold_fpr,
    )
    plot_firewall_results(config)
    update_firewall_summary_markdown(config, f"{config['paths']['results_dir']}/summary.md")
    LOGGER.info("Adversarial Firewall pipeline completed successfully.")


def main() -> None:
    setup_logging()
    try:
        run()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        LOGGER.error("Firewall pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    sys.exit(main())
