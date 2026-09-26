"""Train the Fashion-MNIST model and inspect its first validation predictions."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from script2_dataloaders import create_dataloaders
from script3_model import device, model
from script4_train import train_model


NUMBER_OF_PREDICTIONS = 3
NUMBER_OF_TOP_CLASSES = 4


def count_parameters(model_to_count: nn.Module) -> int:
    """Return the total number of trainable and non-trainable parameters."""
    return sum(parameter.numel() for parameter in model_to_count.parameters())


def show_predictions(
    trained_model: nn.Module,
    validation_loader: DataLoader,
    *,
    prediction_device: torch.device = device,
) -> None:
    """Print detailed predictions for the first three validation images."""
    class_names = validation_loader.dataset.dataset.classes
    images, labels = next(iter(validation_loader))
    images = images[:NUMBER_OF_PREDICTIONS].to(prediction_device)
    labels = labels[:NUMBER_OF_PREDICTIONS]

    trained_model.to(prediction_device)
    trained_model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(trained_model(images), dim=1).cpu()

    print(f"Total model parameters: {count_parameters(trained_model):,}")

    for image_number, (label, class_probabilities) in enumerate(
        zip(labels, probabilities), start=1
    ):
        predicted_index = class_probabilities.argmax().item()
        actual_index = label.item()
        print(f"\nValidation image {image_number}")
        print(f"Predicted: {class_names[predicted_index]}")
        print(f"Actual:    {class_names[actual_index]}")

        print("Probability for each class:")
        for class_name, probability in zip(class_names, class_probabilities):
            print(f"  {class_name:<12} {probability.item():.2%}")

        top_probabilities, top_indices = torch.topk(
            class_probabilities, NUMBER_OF_TOP_CLASSES
        )
        print(f"Top {NUMBER_OF_TOP_CLASSES} most likely classes:")
        for rank, (class_index, probability) in enumerate(
            zip(top_indices, top_probabilities), start=1
        ):
            print(
                f"  {rank}. {class_names[class_index.item()]:<12} "
                f"{probability.item():.2%}"
            )


if __name__ == "__main__":
    training_dataloader, validation_dataloader, _ = create_dataloaders()
    train_model(model, training_dataloader, validation_dataloader)
    show_predictions(model, validation_dataloader)
