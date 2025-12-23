
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/width_weight_decay_results.csv")

    weight_decays = sorted(df["weight_decay"].unique())

    # Plot test accuracy vs width
    plt.figure()
    for wd in weight_decays:
        subset = df[df["weight_decay"] == wd]
        plt.plot(
            subset["width"],
            subset["test_accuracy"],
            marker="o",
            label=f"weight_decay={wd}"
        )

    plt.xlabel("Network Width")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Network Width")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/test_accuracy_vs_width.png")
    plt.close()

    # Plot generalization gap vs width
    plt.figure()
    for wd in weight_decays:
        subset = df[df["weight_decay"] == wd]
        plt.plot(
            subset["width"],
            subset["generalization_gap"],
            marker="o",
            label=f"weight_decay={wd}"
        )

    plt.xlabel("Network Width")
    plt.ylabel("Generalization Gap")
    plt.title("Generalization Gap vs Network Width")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/generalization_gap_vs_width.png")
    plt.close()


if __name__ == "__main__":
    main()


