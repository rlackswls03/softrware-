"""MNIST data loading and deterministic splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from adversarial_mnist.utils import make_generator, safe_num_workers, seed_worker


@dataclass(frozen=True)
class DataLoaders:
    """Container for train, validation, and test DataLoaders."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_size: int
    validation_size: int
    test_size: int


def mnist_transform() -> transforms.Compose:
    """Return the default MNIST transform; no normalization is applied."""
    return transforms.Compose([transforms.ToTensor()])


def _deterministic_indices(length: int, seed: int) -> list[int]:
    generator = make_generator(seed)
    return torch.randperm(length, generator=generator).tolist()


def _subset_by_count(dataset: Dataset, count: int | None, seed: int | None = None) -> Dataset:
    if count is None:
        return dataset
    if count < 0:
        raise ValueError("Subset count must be non-negative.")
    if count > len(dataset):
        raise ValueError(f"Requested subset count {count} exceeds dataset length {len(dataset)}.")
    indices = list(range(len(dataset))) if seed is None else _deterministic_indices(len(dataset), seed)
    return Subset(dataset, indices[:count])


def create_train_validation_split(
    dataset: Dataset,
    validation_size: int,
    seed: int,
    train_subset: int | None = None,
    validation_subset: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Split a training dataset into deterministic train and validation subsets."""
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError("validation_size must be between 1 and len(dataset)-1.")
    indices = _deterministic_indices(len(dataset), seed)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    if train_subset is not None:
        if train_subset > len(train_indices):
            raise ValueError("train_subset exceeds available training samples.")
        train_indices = train_indices[:train_subset]
    if validation_subset is not None:
        if validation_subset > len(validation_indices):
            raise ValueError("validation_subset exceeds available validation samples.")
        validation_indices = validation_indices[:validation_subset]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a seed-aware DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )


def create_mnist_dataloaders(
    config: dict[str, Any],
    seed: int,
) -> DataLoaders:
    """Download/load MNIST and return deterministic train/validation/test loaders."""
    dataset_config = config["dataset"]
    training_config = config["training"]
    data_dir = Path(dataset_config.get("data_dir", "data"))
    transform = mnist_transform()

    train_full = datasets.MNIST(
        root=data_dir,
        train=True,
        download=bool(dataset_config.get("download", True)),
        transform=transform,
    )
    test_full = datasets.MNIST(
        root=data_dir,
        train=False,
        download=bool(dataset_config.get("download", True)),
        transform=transform,
    )

    train_dataset, validation_dataset = create_train_validation_split(
        train_full,
        validation_size=int(dataset_config.get("validation_size", 5000)),
        seed=seed,
        train_subset=dataset_config.get("train_subset"),
        validation_subset=dataset_config.get("validation_subset"),
    )
    fixed_test_seed = int(dataset_config.get("fixed_test_subset_seed", seed))
    test_dataset = _subset_by_count(
        test_full,
        dataset_config.get("test_subset"),
        seed=fixed_test_seed,
    )

    num_workers = safe_num_workers(training_config.get("num_workers"))
    return DataLoaders(
        train=make_dataloader(
            train_dataset,
            batch_size=int(training_config["batch_size"]),
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
        ),
        validation=make_dataloader(
            validation_dataset,
            batch_size=int(training_config["test_batch_size"]),
            seed=seed + 1,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=make_dataloader(
            test_dataset,
            batch_size=int(training_config["test_batch_size"]),
            seed=seed + 2,
            shuffle=False,
            num_workers=num_workers,
        ),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        test_size=len(test_dataset),
    )


class TensorDataset(Dataset):
    """Small tensor dataset used by smoke tests."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have matching first dimension.")
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def create_synthetic_dataloaders(seed: int = 42, batch_size: int = 8) -> DataLoaders:
    """Create tiny deterministic synthetic loaders for smoke testing."""
    generator = make_generator(seed)
    train_images = torch.rand((32, 1, 28, 28), generator=generator)
    train_labels = torch.randint(0, 10, (32,), generator=generator)
    val_images = torch.rand((16, 1, 28, 28), generator=generator)
    val_labels = torch.randint(0, 10, (16,), generator=generator)
    test_images = torch.rand((16, 1, 28, 28), generator=generator)
    test_labels = torch.randint(0, 10, (16,), generator=generator)
    num_workers = 0
    return DataLoaders(
        train=make_dataloader(
            TensorDataset(train_images, train_labels),
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=num_workers,
        ),
        validation=make_dataloader(
            TensorDataset(val_images, val_labels),
            batch_size=batch_size,
            seed=seed + 1,
            shuffle=False,
            num_workers=num_workers,
        ),
        test=make_dataloader(
            TensorDataset(test_images, test_labels),
            batch_size=batch_size,
            seed=seed + 2,
            shuffle=False,
            num_workers=num_workers,
        ),
        train_size=32,
        validation_size=16,
        test_size=16,
    )
