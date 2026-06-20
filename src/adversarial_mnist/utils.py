"""Shared utility functions for reproducible experiments."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for CLI scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return data


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Save a JSON object with stable formatting."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` in place and return it."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_quick_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply quick-mode overrides to an experiment config."""
    quick = config.get("quick", {})
    if not isinstance(quick, dict):
        raise ValueError("Config field 'quick' must be an object.")
    config["quick"]["enabled"] = True
    if "seeds" in quick:
        config["seeds"] = quick["seeds"]
    for section in ("dataset", "training", "evaluation"):
        overrides = quick.get(section)
        if isinstance(overrides, Mapping):
            deep_update(config[section], overrides)
    return config


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, Torch, CUDA, and deterministic backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a CPU torch generator seeded for DataLoader sampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def safe_num_workers(config_value: int | None = None) -> int:
    """Choose a conservative DataLoader worker count."""
    if config_value is not None:
        if config_value < 0:
            raise ValueError("num_workers must be non-negative.")
        return config_value
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(4, max(0, cpu_count - 1))


def choose_device(preferred: str | None = None) -> torch.device:
    """Select CUDA, MPS, or CPU unless an available preferred device is given."""
    if preferred:
        normalized = preferred.lower()
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if normalized == "mps" and (
            not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested but is not available.")
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def package_version(package_name: str) -> str | None:
    """Return an installed package version or ``None``."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_git_commit(repo_dir: str | Path = ".") -> str | None:
    """Return current Git commit hash, or ``None`` if unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_environment(device: torch.device | None = None, repo_dir: str | Path = ".") -> dict[str, Any]:
    """Collect runtime environment metadata."""
    selected_device = device or choose_device()
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
            )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "numpy": np.__version__,
        "pandas": package_version("pandas"),
        "matplotlib": package_version("matplotlib"),
        "device": str(selected_device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "git_commit": get_git_commit(repo_dir),
    }


def write_environment(path: str | Path, device: torch.device | None = None) -> None:
    """Write environment metadata to JSON."""
    save_json(collect_environment(device=device), path)


def append_csv_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Append rows to a CSV file, creating parent directories as needed."""
    import pandas as pd

    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = pd.DataFrame(rows)
    if output_path.exists():
        frame.to_csv(output_path, mode="a", index=False, header=False)
    else:
        frame.to_csv(output_path, index=False)
