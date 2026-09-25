"""Load Fashion-MNIST and create reproducible training and validation splits."""

import torch
from torch.utils.data import random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


RANDOM_SEED = 42
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000


def load_datasets(data_dir: str = "data"):
    """Download Fashion-MNIST and return its train, validation, and test datasets."""
    transform = ToTensor()

    full_training_set = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    training_set, validation_set = random_split(
        full_training_set,
        [TRAIN_SIZE, VALIDATION_SIZE],
        generator=generator,
    )
    return training_set, validation_set, test_set


if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = load_datasets()
    print(f"Training images: {len(train_dataset):,}")
    print(f"Validation images: {len(validation_dataset):,}")
    print(f"Test images: {len(test_dataset):,}")
