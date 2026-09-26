"""Train the Fashion-MNIST classifier and retain epoch-by-epoch metrics."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from script2_dataloaders import create_dataloaders
from script3_model import device, model


EPOCHS = 20
LEARNING_RATE = 0.1
NUMBER_OF_CLASSES = 10


def train_model(
    model_to_train: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    training_device: torch.device = device,
) -> dict[str, list[float]]:
    """Train a model and return its loss and accuracy history by epoch."""
    model_to_train.to(training_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_to_train.parameters(), lr=learning_rate)
    training_accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(
        training_device
    )
    validation_accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(
        training_device
    )
    history: dict[str, list[float]] = {
        "loss": [],
        "training_accuracy": [],
        "validation_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        model_to_train.train()
        total_loss = 0.0
        total_examples = 0

        for images, labels in training_loader:
            images = images.to(training_device)
            labels = labels.to(training_device)

            optimizer.zero_grad()
            logits = model_to_train(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            training_accuracy.update(logits.detach(), labels)

        epoch_loss = total_loss / total_examples
        epoch_training_accuracy = training_accuracy.compute().item()
        training_accuracy.reset()

        model_to_train.eval()
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(training_device)
                labels = labels.to(training_device)
                logits = model_to_train(images)
                validation_accuracy.update(logits, labels)

        epoch_validation_accuracy = validation_accuracy.compute().item()
        validation_accuracy.reset()

        history["loss"].append(epoch_loss)
        history["training_accuracy"].append(epoch_training_accuracy)
        history["validation_accuracy"].append(epoch_validation_accuracy)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Training accuracy: {epoch_training_accuracy:.4f} | "
            f"Validation accuracy: {epoch_validation_accuracy:.4f}"
        )

    return history


if __name__ == "__main__":
    train_dataloader, validation_dataloader, _ = create_dataloaders()
    history = train_model(model, train_dataloader, validation_dataloader)
