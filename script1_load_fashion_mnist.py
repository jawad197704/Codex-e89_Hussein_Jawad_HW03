"""Download Fashion-MNIST and create reproducible dataset splits."""

from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


DATASET_DIRECTORY = Path(__file__).resolve().parent / "datasets"
RANDOM_SEED = 42
TRAINING_SIZE = 55_000
VALIDATION_SIZE = 5_000


def load_fashion_mnist() -> tuple[Dataset, Dataset, Dataset]:
    """Return the Fashion-MNIST training, validation, and test datasets."""
    # ToTensor converts each image to float32 and scales uint8 pixels to [0, 1].
    transform = ToTensor()

    complete_training_dataset = FashionMNIST(
        root=DATASET_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = FashionMNIST(
        root=DATASET_DIRECTORY,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    training_dataset, validation_dataset = random_split(
        complete_training_dataset,
        [TRAINING_SIZE, VALIDATION_SIZE],
        generator=generator,
    )

    return training_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    training_dataset, validation_dataset, test_dataset = load_fashion_mnist()
    print(f"Training samples: {len(training_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
