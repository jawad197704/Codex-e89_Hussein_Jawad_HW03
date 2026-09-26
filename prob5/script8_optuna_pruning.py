"""Tune the Fashion-MNIST classifier while pruning unpromising trials."""

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from script2_dataloaders import create_dataloaders
from script3_model import device
from script7_optuna import RANDOM_SEED, TunableFashionMNISTClassifier


NUMBER_OF_TRIALS = 20
EPOCHS_PER_TRIAL = 10


def objective(
    trial: optuna.Trial,
    training_loader: DataLoader,
    validation_loader: DataLoader,
) -> float:
    """Train a sampled model, reporting accuracy after every epoch for pruning."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_neurons = trial.suggest_int("hidden_neurons", 20, 300)

    # Use the same initialization for each trial so parameter choices, rather than
    # random starting weights, drive differences between trials.
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model = TunableFashionMNISTClassifier(hidden_neurons).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_validation_accuracy = 0.0

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct_predictions = 0
        total_examples = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_examples += labels.size(0)

        validation_accuracy = correct_predictions / total_examples
        best_validation_accuracy = max(
            best_validation_accuracy, validation_accuracy
        )
        print(
            f"Trial {trial.number:02d} | Epoch {epoch + 1:02d}/"
            f"{EPOCHS_PER_TRIAL} | Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        trial.report(validation_accuracy, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at epoch {epoch + 1} with validation accuracy "
                f"{validation_accuracy:.4f}"
            )

    return best_validation_accuracy


def run_study() -> optuna.Study:
    """Run 20 trials with median pruning and print the best result."""
    training_loader, validation_loader, _ = create_dataloaders()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective(trial, training_loader, validation_loader),
        n_trials=NUMBER_OF_TRIALS,
    )

    print("\nBest parameters:")
    print(f"  Learning rate: {study.best_params['learning_rate']:.6g}")
    print(f"  Hidden neurons per layer: {study.best_params['hidden_neurons']}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    return study


if __name__ == "__main__":
    run_study()
