
import csv
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

    widths = [64, 128, 256, 512, 1024]
    weight_decays = [0.0, 1e-4, 1e-3]
    epochs = 20

    results_file = "results/width_weight_decay_results.csv"

    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "width",
            "weight_decay",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "generalization_gap"
        ])

        for wd in weight_decays:
            for width in widths:
                print(f"Running: width={width}, weight_decay={wd}")

                model = MLP(hidden_dim=width).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=1e-3,
                    weight_decay=wd
                )

                for _ in range(epochs):
                    train_one_epoch(
                        model, train_loader, optimizer, criterion, device
                    )

                train_loss, train_acc = evaluate(
                    model, train_loader, criterion, device
                )
                test_loss, test_acc = evaluate(
                    model, test_loader, criterion, device
                )

                gap = train_acc - test_acc

                writer.writerow([
                    width,
                    wd,
                    train_loss,
                    train_acc,
                    test_loss,
                    test_acc,
                    gap
                ])


if __name__ == "__main__":
    main()
