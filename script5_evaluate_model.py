"""Train the Fashion-MNIST classifier and inspect validation predictions."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from script2_inspect_fashion_mnist_data import create_data_loaders
from script3_image_classifier import device, loss_function, model
from script4_train_model import (
    LEARNING_RATE,
    NUMBER_OF_CLASSES,
    NUMBER_OF_EPOCHS,
    train_model,
)


FOURTH_IMAGE_INDEX = 3
NUMBER_OF_TOP_CLASSES = 4


def count_parameters(model_to_count: nn.Module) -> int:
    """Return the model's total number of trainable and fixed parameters."""
    return sum(parameter.numel() for parameter in model_to_count.parameters())


def evaluate_model(
    trained_model: nn.Module,
    validation_loader: DataLoader,
    evaluation_device: torch.device = device,
) -> None:
    """Run inference on one validation batch and display its predictions."""
    class_names = validation_loader.dataset.dataset.classes
    images, _ = next(iter(validation_loader))
    images = images.to(evaluation_device)

    trained_model.to(evaluation_device)
    trained_model.eval()
    with torch.no_grad():
        logits = trained_model(images)
        predicted_indices = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)

    predicted_class_names = [
        class_names[class_index.item()] for class_index in predicted_indices
    ]
    print(f"Fourth image predicted class: {predicted_class_names[FOURTH_IMAGE_INDEX]}")

    # MPS tensors are moved to the CPU before formatting and displaying them.
    if evaluation_device.type == "mps":
        probabilities = probabilities.cpu()
        predicted_indices = predicted_indices.cpu()

    rounded_probabilities = probabilities.round(decimals=3)
    for image_number, (image_probabilities, rounded_image_probabilities) in enumerate(
        zip(probabilities, rounded_probabilities), start=1
    ):
        print(f"\nImage {image_number}")
        predicted_index = predicted_indices[image_number - 1].item()
        print(f"Predicted class: {class_names[predicted_index]}")
        print(f"Softmax probabilities: {rounded_image_probabilities}")

        top_probabilities, top_indices = torch.topk(
            image_probabilities, NUMBER_OF_TOP_CLASSES
        )
        print(f"Top {NUMBER_OF_TOP_CLASSES} classes:")
        for class_index, probability in zip(top_indices, top_probabilities):
            print(
                f"  {class_names[class_index.item()]}: "
                f"{probability.item():.3f}"
            )

    print(f"\nTotal model parameters: {count_parameters(trained_model):,}")


if __name__ == "__main__":
    training_loader, validation_loader, _ = create_data_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(device)

    train_model(
        model,
        optimizer,
        loss_function,
        accuracy,
        training_loader,
        validation_loader,
        NUMBER_OF_EPOCHS,
    )
    evaluate_model(model, validation_loader)
