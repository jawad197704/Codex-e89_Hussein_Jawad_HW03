"""Tune the Fashion-MNIST classifier with Optuna."""

import optuna
import torch
from torchmetrics.classification import MulticlassAccuracy

from script2_inspect_fashion_mnist_data import create_data_loaders
from script3_image_classifier import (
    NUMBER_OF_CLASSES,
    FashionMNISTClassifier,
    device,
    loss_function,
)
from script4_train_model import train_model


RANDOM_SEED = 42
NUMBER_OF_EPOCHS = 10
NUMBER_OF_TRIALS = 5


def objective(
    trial: optuna.Trial,
    training_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
) -> float:
    """Train one sampled configuration and return its best validation accuracy."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_layer_size = trial.suggest_int("hidden_layer_size", 20, 300)

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    trial_model = FashionMNISTClassifier(hidden_layer_size=hidden_layer_size).to(device)
    optimizer = torch.optim.SGD(trial_model.parameters(), lr=learning_rate)
    accuracy = MulticlassAccuracy(num_classes=NUMBER_OF_CLASSES).to(device)
    history = train_model(
        trial_model,
        optimizer,
        loss_function,
        accuracy,
        training_loader,
        validation_loader,
        NUMBER_OF_EPOCHS,
    )

    return max(history["validation_accuracy"])


def run_study() -> optuna.Study:
    """Run the seeded hyperparameter search and display its best result."""
    training_loader, validation_loader, _ = create_data_loaders()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, training_loader, validation_loader),
        n_trials=NUMBER_OF_TRIALS,
    )

    print("\nBest parameters:")
    print(f"  Learning rate: {study.best_params['learning_rate']:.6g}")
    print(f"  Hidden-layer size: {study.best_params['hidden_layer_size']}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    return study


if __name__ == "__main__":
    run_study()
