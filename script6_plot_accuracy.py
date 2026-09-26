"""Train the Fashion-MNIST classifier and plot its accuracy by epoch."""

import matplotlib.pyplot as plt

from script2_dataloaders import create_dataloaders
from script3_model import model
from script4_train import train_model


def plot_accuracy(history: dict[str, list[float]]) -> None:
    """Plot the training and validation accuracy stored in a training history."""
    epochs = range(1, len(history["training_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["training_accuracy"], label="Training accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Fashion-MNIST Training Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_dataloader, validation_dataloader, _ = create_dataloaders()
    training_history = train_model(
        model,
        training_dataloader,
        validation_dataloader,
    )
    plot_accuracy(training_history)
