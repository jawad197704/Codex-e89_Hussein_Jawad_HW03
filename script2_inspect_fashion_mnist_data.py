"""Build Fashion-MNIST data loaders and inspect the first training sample."""

import torch
from torch.utils.data import DataLoader

from script1_load_fashion_mnist import load_fashion_mnist


BATCH_SIZE = 32
RANDOM_SEED = 42


def create_data_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return reproducible training, validation, and test data loaders."""
    torch.manual_seed(RANDOM_SEED)
    training_dataset, validation_dataset, test_dataset = load_fashion_mnist()

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return training_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = create_data_loaders()

    image, target = train_loader.dataset[0]
    class_names = train_loader.dataset.dataset.classes

    print(f"Image tensor shape: {image.shape}")
    print(f"Image tensor data type: {image.dtype}")
    print(f"Numeric target: {target}")
    print(f"Class name: {class_names[target]}")
