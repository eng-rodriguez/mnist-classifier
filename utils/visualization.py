import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_metrics(
    trn_losses: List[float],
    tnr_corrects: List[int],
    val_losses: List[float],
    val_corrects: List[int],
    save_path: str = None,
) -> None:
    "Plot training and validation metrics."
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot losses
    ax1.plot(trn_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_title("Losses vs. Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracies
    trn_acc = [t / 600 for t in tnr_corrects]
    val_acc = [t / 100 for t in val_corrects]
    ax2.plot(trn_acc, label="Training Accuracy")
    ax2.plot(val_acc, label="Validation Accuracy")
    ax2.set_title("Accuracy vs. Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
