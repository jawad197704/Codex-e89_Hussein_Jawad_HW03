"""Create DataLoaders and inspect a sample from the Fashion-MNIST data."""

from torch.utils.data import DataLoader

from script1_load_dataset import load_datasets


BATCH_SIZE = 32


def create_dataloaders(data_dir: str = "data"):
    """Return training, validation, and test DataLoaders for Fashion-MNIST."""
    training_set, validation_set, test_set = load_datasets(data_dir)

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


if __name__ == "__main__":
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders()

    image, label = train_dataloader.dataset[0]
    class_names = train_dataloader.dataset.dataset.classes

    print(f"First training sample shape: {image.shape}")
    print(f"First training sample data type: {image.dtype}")
    print(f"First training sample class name: {class_names[label]}")
