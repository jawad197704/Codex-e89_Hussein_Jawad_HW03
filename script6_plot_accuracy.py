"""Plot accuracy values from an existing Fashion-MNIST training history."""

import matplotlib.pyplot as plt


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
