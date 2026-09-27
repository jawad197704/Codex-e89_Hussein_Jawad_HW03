"""Plot training accuracy from an existing Fashion-MNIST history."""

import matplotlib.pyplot as plt


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
