"""Run the full MNIST adversarial robustness pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from adversarial_mnist.data import create_mnist_dataloaders
from adversarial_mnist.evaluation import (
    aggregate_result_files,
    evaluate_all_clean,
    evaluate_all_fgsm_robustness,
    evaluate_all_pgd_whitebox,
    evaluate_transferability,
    load_trained_models,
    save_adversarial_example_batch,
    write_csv,
)
from adversarial_mnist.training import train_model
from adversarial_mnist.utils import (
    save_json,
    set_reproducibility,
    setup_logging,
    write_environment,
)
from adversarial_mnist.visualization import generate_summary_markdown, plot_all_results
from scripts.common import (
    add_common_args,
    prepare_config,
    prepare_device,
    remove_generated_results,
    save_run_config,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def run() -> None:
    args = build_parser().parse_args()
    config = prepare_config(args)
    if args.force:
        remove_generated_results(config)
    device = prepare_device(args)
    write_environment(f"{config['paths']['results_dir']}/environment.json", device=device)
    save_run_config(config)

    if config.get("quick", {}).get("enabled"):
        LOGGER.warning(
            "Quick mode is only for pipeline connectivity checks; do not draw performance conclusions."
        )

    training_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    fgsm_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    pgd_rows: list[dict[str, object]] = []

    first_models = None
    first_loader = None
    for seed in config["seeds"]:
        LOGGER.info("Starting seed %s", seed)
        set_reproducibility(int(seed), deterministic=bool(config["training"]["deterministic"]))
        loaders = create_mnist_dataloaders(config, seed=int(seed))

        for model_key in config["models"]:
            _, rows = train_model(
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
            training_rows.extend(rows)

        models = load_trained_models(
            config["models"],
            config["paths"]["checkpoints_dir"],
            seed=int(seed),
            device=device,
        )
        if first_models is None:
            first_models = models
            first_loader = loaders.test

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
        transfer_rows.extend(
            evaluate_transferability(
                models,
                loaders.test,
                config["evaluation"]["fgsm_epsilons"],
                int(seed),
                device,
                model_order=config["evaluation"]["transfer_model_order"],
            )
        )
        pgd_rows.extend(
            evaluate_all_pgd_whitebox(
                models,
                loaders.test,
                int(seed),
                device,
                config["evaluation"]["pgd"],
            )
        )

    raw_dir = config["paths"]["raw_dir"]
    if training_rows:
        write_csv(training_rows, f"{raw_dir}/training_history.csv")
    else:
        save_json(
            {"message": "No new training rows were produced; existing checkpoints may have been reused."},
            f"{raw_dir}/training_history_note.json",
        )
    write_csv(clean_rows, f"{raw_dir}/clean_accuracy.csv")
    write_csv(fgsm_rows, f"{raw_dir}/fgsm_robustness.csv")
    write_csv(transfer_rows, f"{raw_dir}/transferability_long.csv")
    write_csv(pgd_rows, f"{raw_dir}/pgd_whitebox.csv")

    if first_models is None or first_loader is None:
        raise RuntimeError("No model was available for adversarial example plotting.")
    save_adversarial_example_batch(
        first_models[config["models"][0]],
        first_loader,
        f"{raw_dir}/adversarial_examples.pt",
        device=device,
        epsilon=float(config["evaluation"]["pgd"]["epsilon"]),
        pgd_steps=int(config["evaluation"]["pgd"]["steps"]),
    )

    aggregate_result_files(config["paths"]["raw_dir"], config["paths"]["aggregated_dir"])
    plot_all_results(config)
    generate_summary_markdown(config, f"{config['paths']['results_dir']}/summary.md")
    LOGGER.info("Pipeline completed successfully.")


def main() -> None:
    setup_logging()
    try:
        run()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    sys.exit(main())
