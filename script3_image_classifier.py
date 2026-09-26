"""Define a fully connected classifier for Fashion-MNIST images."""

import torch
from torch import nn


RANDOM_SEED = 42
INPUT_FEATURES = 28 * 28
NUMBER_OF_CLASSES = 10


class FashionMNISTClassifier(nn.Module):
    """Classify 28-by-28 grayscale images into 10 clothing categories."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, NUMBER_OF_CLASSES),
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
