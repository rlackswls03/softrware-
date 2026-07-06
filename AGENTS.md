# AGENTS.md

## Purpose

This repository studies MNIST adversarial robustness with FGSM, FGSM adversarial
training, transfer attacks, PGD white-box evaluation, and an Adversarial Firewall
prototype for input detection, purification, and rejection.

## Structure

- `src/adversarial_mnist/`: reusable package code.
- `scripts/`: command-line entry points for training, evaluation, plotting, and smoke tests.
- `configs/default.json`: default experiment configuration.
- `tests/`: unit tests that do not require MNIST downloads.
- `results/`, `checkpoints/`, `data/`: generated artifacts and local datasets.

## Commands

```bash
python -m pip install -e .
python -m ruff check .
python -m pytest -q
python -m scripts.smoke_test
python -m scripts.run_pipeline --quick
python -m scripts.evaluate_pgd_strong --seeds 42 123 2026 --test-subset 10000 --device cuda --force
python -m scripts.evaluate_firewall --seeds 42 123 2026 --force
```

## Ground Rules

- Do not fabricate or massage experiment results.
- Never train on the MNIST test set.
- Epsilon values are defined in `[0, 1]` pixel space; the default transform is `ToTensor()`.
- After code changes, run `python -m pytest`, `python -m ruff check .`, and the smoke test when the environment allows.
- Document only numbers produced by real executions.
- Keep code typed, modular, and import-safe; importing modules must not download data or start training.
