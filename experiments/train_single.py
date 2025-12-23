
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mlp import MLP
from utils.train import train_one_epoch
from utils.evaluate import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model = MLP(hidden_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 5  # keep small for sanity check

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}: "
            f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
