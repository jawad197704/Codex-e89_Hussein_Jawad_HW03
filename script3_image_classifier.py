"""Define a fully connected classifier for Fashion-MNIST images."""

import torch
from torch import nn


RANDOM_SEED = 42
INPUT_FEATURES = 28 * 28
NUMBER_OF_CLASSES = 10


class FashionMNISTClassifier(nn.Module):
    """Classify 28-by-28 grayscale images into 10 clothing categories."""

    def __init__(self, hidden_layer_size: int | None = None) -> None:
        super().__init__()
        first_hidden_size = 300 if hidden_layer_size is None else hidden_layer_size
        second_hidden_size = 100 if hidden_layer_size is None else hidden_layer_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, first_hidden_size),
            nn.ReLU(),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.ReLU(),
            nn.Linear(second_hidden_size, NUMBER_OF_CLASSES),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of Fashion-MNIST images."""
        return self.network(images)


torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTClassifier().to(device)
loss_function = nn.CrossEntropyLoss()


if __name__ == "__main__":
    print(model)
    print(f"Using device: {device}")
    print(f"Loss function: {loss_function}")
