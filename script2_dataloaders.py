"""Create CIFAR-10 data loaders and inspect the first training sample."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


BATCH_SIZE = 32
VALIDATION_SIZE = 5_000
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"


def create_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return training, validation, and test loaders for CIFAR-10."""
    transform = transforms.ToTensor()
    complete_training_set = datasets.CIFAR10(
        root=DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.CIFAR10(
        root=DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transform,
    )

    training_size = len(complete_training_set) - VALIDATION_SIZE
    training_set, validation_set = random_split(
        complete_training_set,
        [training_size, VALIDATION_SIZE],
        generator=torch.Generator().manual_seed(42),
    )

    training_loader = DataLoader(
        training_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return training_loader, validation_loader, test_loader


def main() -> None:
    training_loader, validation_loader, test_loader = create_dataloaders()

    # Index the underlying training subset so that the reported item is the
    # first sample, independent of the loader's intentionally shuffled order.
    image, label = training_loader.dataset[0]
    class_name = training_loader.dataset.dataset.classes[label]

    print(f"Training batches: {len(training_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"First training sample shape: {image.shape}")
    print(f"First training sample data type: {image.dtype}")
    print(f"First training sample class name: {class_name}")


if __name__ == "__main__":
    main()
