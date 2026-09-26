"""Tune the Fashion-MNIST classifier's learning rate and hidden-layer size."""

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from script2_dataloaders import create_dataloaders
from script3_model import INPUT_FEATURES, NUMBER_OF_CLASSES, device
from script4_train import train_model


NUMBER_OF_TRIALS = 5
EPOCHS_PER_TRIAL = 10
RANDOM_SEED = 42


class TunableFashionMNISTClassifier(nn.Module):
    """A classifier whose two hidden layers have the same tunable width."""

    def __init__(self, hidden_neurons: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, NUMBER_OF_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of Fashion-MNIST images."""
        return self.network(images)


def objective(
    trial: optuna.Trial,
    training_loader: DataLoader,
    validation_loader: DataLoader,
) -> float:
    """Train one parameter configuration and return its best validation accuracy."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_neurons = trial.suggest_int("hidden_neurons", 20, 300)

    # Give every trial the same initialization and shuffled-batch sequence so that
    # differences in validation accuracy come from the sampled hyperparameters.
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    trial_model = TunableFashionMNISTClassifier(hidden_neurons)
    history = train_model(
        trial_model,
        training_loader,
        validation_loader,
        epochs=EPOCHS_PER_TRIAL,
        learning_rate=learning_rate,
        training_device=device,
    )
    return max(history["validation_accuracy"])


def run_study() -> optuna.Study:
    """Run five reproducible trials and print the best result."""
    training_loader, validation_loader, _ = create_dataloaders()
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
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
