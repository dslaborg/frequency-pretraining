"""
This script creates Figure 4 in the paper. It visualizes the training and validation loss, hamming metric, and frequency
bin accuracy for the pretraining phase of the model. The results are extracted from a single pretraining run performed
in exp001b.
"""
import ast
import json

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("white")


def plot_metrics(results: dict, freq_gt: np.ndarray, freq_pred: np.ndarray):
    # read metrics from results.json
    train_metrics = results["train"]["metrics"]["freq"]
    train_pretraining_metrics = results["train_pretraining"]["metrics"]["freq"]
    valid_metrics = results["valid_pretraining"]["metrics"]["freq"]
    n_epochs = len(train_metrics["acc_freq"])

    f, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes = axes.flatten()

    # first panel shows the loss curves on the training and validation sets
    axes[0].plot(
        np.arange(n_epochs + 1),
        train_pretraining_metrics["loss"],
        "--",
        label="Train data",
    )
    axes[0].plot(
        np.arange(n_epochs + 1), valid_metrics["loss"], "-", label="Validation data"
    )
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlabel("Training epochs")
    axes[0].set_ylabel("Cross-entropy loss")

    # second panel shows the hamming metric on the training and validation sets
    axes[1].plot(
        np.arange(n_epochs + 1),
        [l for l in train_pretraining_metrics["hamming_freq"]],
        "--",
        label="Train data",
    )
    axes[1].plot(
        np.arange(n_epochs + 1),
        [l for l in valid_metrics["hamming_freq"]],
        "-",
        label="Validation data",
    )
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel("Training epochs")
    axes[1].set_ylabel("Hamming metric")

    # third panel shows the accuracy for each frequency bin after the last pretraining epoch
    freq_scores = []
    for c in range(freq_gt.shape[1]):
        freq_scores.append(np.mean(freq_gt[:, c] == freq_pred[:, c]))

    # the frequency bins are a base 2 logarithmic scale from 0.3 to 35 Hz
    # code to generate the frequency bins:
    # [round(f, 2) for f in 2 ** np.linspace(np.log2(0.3), np.log2(35), 21)]
    freq_ticks = [
        "0.30",
        "0.38",
        "0.48",
        "0.61",
        "0.78",
        "0.99",
        "1.25",
        "1.59",
        "2.01",
        "2.55",
        "3.24",
        "4.11",
        "5.22",
        "6.62",
        "8.39",
        "10.65",
        "13.51",
        "17.14",
        "21.75",
        "27.59",
        "35.00",
    ]
    axes[2].bar(np.arange(len(freq_scores)), freq_scores)
    axes[2].set_xticks(
        np.arange(len(freq_scores) + 1) - 0.5,
        freq_ticks,
        rotation=60,
        ha="right",
        rotation_mode="anchor",
    )
    axes[2].grid()
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.5, 1.05)

    plt.text(0.48, 1.03, "a", transform=axes[0].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "b", transform=axes[1].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "c", transform=axes[2].transAxes, size=12, weight="bold")


def main():
    # relative path to the parent folder of the experiment folders that contain the results to be visualized
    result_folder = "../../logs/exp001/exp001b/2023-11-28_14-13-48"
    result_file = f"{result_folder}/results.json"
    with open(result_file) as f:
        results = json.load(f)

    # files with the ground truth and predicted frequency bins
    gt_freq_file = f"{result_folder}/valid_pretraining_freq_gt.txt"
    pred_freq_file = f"{result_folder}/valid_pretraining_freq_pred_20.txt"
    with open(gt_freq_file) as f:
        gt_freq = np.array([ast.literal_eval(line) for line in f.readlines()])
    with open(pred_freq_file) as f:
        pred_freq = np.array([ast.literal_eval(line) for line in f.readlines()])

    plot_metrics(results, gt_freq, pred_freq)
    plt.show()


if __name__ == "__main__":
    main()
