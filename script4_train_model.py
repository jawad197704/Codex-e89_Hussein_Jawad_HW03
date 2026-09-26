"""Configure and run training for the Fashion-MNIST classifier."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from script2_inspect_fashion_mnist_data import create_data_loaders
from script3_image_classifier import device, loss_function, model


LEARNING_RATE = 0.1
NUMBER_OF_CLASSES = 10
NUMBER_OF_EPOCHS = 10


def train_model(
    model_to_train: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    accuracy_metric: MulticlassAccuracy,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
) -> dict[str, list[float]]:
    """Train a model and return its loss and accuracy history by epoch."""
    history: dict[str, list[float]] = {
        "loss": [],
        "training_accuracy": [],
        "validation_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model_to_train.train()
        training_loss = 0.0

        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model_to_train(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * labels.size(0)
            accuracy_metric.update(logits.detach(), labels)

        average_training_loss = training_loss / len(training_loader.dataset)
        training_accuracy = accuracy_metric.compute().item()
        accuracy_metric.reset()

        model_to_train.eval()
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                accuracy_metric.update(model_to_train(images), labels)

        validation_accuracy = accuracy_metric.compute().item()
        accuracy_metric.reset()

        history["loss"].append(average_training_loss)
        history["training_accuracy"].append(training_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {average_training_loss:.4f} | "
            f"Training accuracy: {training_accuracy:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

    return history


if __name__ == "__main__":
    train_loader, validation_loader, _ = create_data_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(device)

    train_model(
        model,
        optimizer,
        loss_function,
        accuracy,
        train_loader,
        validation_loader,
        NUMBER_OF_EPOCHS,
    )
