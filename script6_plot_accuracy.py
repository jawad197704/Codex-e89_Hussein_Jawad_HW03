"""Train the Fashion-MNIST classifier and plot its accuracy by epoch."""

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
    """Plot training and validation accuracy from a training history."""
    epochs = range(1, len(history["training_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["training_accuracy"], label="Training accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fashion-MNIST Accuracy by Epoch")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_loader, validation_loader, _ = create_data_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(device)

    training_history = train_model(
        model,
        optimizer,
        loss_function,
        accuracy,
        training_loader,
        validation_loader,
        NUMBER_OF_EPOCHS,
    )
    plot_accuracy(training_history)
