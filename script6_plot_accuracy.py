"""Plot training accuracy from an existing Fashion-MNIST history."""

import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import MulticlassAccuracy

from script2_inspect_fashion_mnist_data import create_data_loaders
from script3_image_classifier import device, loss_function, model
from script4_train_model import (
    LEARNING_RATE,
    NUMBER_OF_CLASSES,
    NUMBER_OF_EPOCHS,
    train_model,
)


def plot_accuracy(history: dict[str, list[float]]) -> None:
    """Plot the training accuracy recorded after each epoch."""
    training_accuracy = history["training_accuracy"]
    epochs = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_accuracy, marker="o", label="Training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title("Fashion-MNIST Training Accuracy by Epoch")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_loader, validation_loader, _ = create_data_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(device)

    history = train_model(
        model,
        optimizer,
        loss_function,
        accuracy,
        training_loader,
        validation_loader,
        NUMBER_OF_EPOCHS,
    )
    plot_accuracy(history)
