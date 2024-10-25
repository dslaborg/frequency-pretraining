"""
This script creates Figure 4 in the paper. It visualizes the training and validation loss, hamming metric, and frequency
bin accuracy for the pretraining phase of the model. The results are extracted from a single pretraining run performed
in exp001b.
The Figure also includes the results of a hyperparmeter exploration in exp007, where the model is pretrained with a
different number of synthetic samples and fine-tuned with the data of 1 subject from the DODO/H dataset.
"""
import ast
import json
import re
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils import parse_result_folder

sns.set_style("white")

patterns = ["//", "--", "xx", "oo"]

expid_to_label = {
    "exp007a_3": "Fixed FE",
    "exp007b_3": "Fine-tuned FE",
}


def plot_metrics(results: dict, freq_gt: np.ndarray, freq_pred: np.ndarray):
    # read metrics from results.json
    train_metrics = results["train"]["metrics"]["freq"]
    train_pretraining_metrics = results["train_pretraining"]["metrics"]["freq"]
    valid_metrics = results["valid_pretraining"]["metrics"]["freq"]
    n_epochs = len(train_metrics["acc_freq"])

    f, axes = plt.subplots(2, 2, figsize=(12, 3.5))
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
    axes[0].xaxis.set_ticks_position("bottom")
    axes[0].tick_params(which="major", width=1.00, length=4)

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
    axes[1].xaxis.set_ticks_position("bottom")
    axes[1].tick_params(which="major", width=1.00, length=4)

    # third panel shows the accuracy for each frequency bin after the last pretraining epoch
    freq_scores = []
    for c in range(freq_gt.shape[1]):
        freq_scores.append(np.mean(freq_gt[:, c] == freq_pred[:, c]))

    # the frequency bins are a base 2 logarithmic scale from 0.3 to 35 Hz
    freq_ticks = [
        f"{round(f, 2):.2f}" for f in 2 ** np.linspace(np.log2(0.3), np.log2(35), 21)
    ]
    axes[2].bar(np.arange(len(freq_scores)), freq_scores)
    axes[2].xaxis.set_ticks_position("bottom")
    axes[2].tick_params(which="major", width=1.00, length=4)
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

    # fourth panel shows the f1 scores for the different number of synthetic samples
    plot_last_panel(axes[3])

    plt.text(0.48, 1.03, "a", transform=axes[0].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "b", transform=axes[1].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "c", transform=axes[2].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "d", transform=axes[3].transAxes, size=12, weight="bold")


def plot_last_panel(ax):
    # get path of __file__
    base_path_of_file = "../../logs/exp007"
    result_folder = join(base_path_of_file, ".")
    dataset = "earlystopping"
    n_subjects = 1

    run_filter = ".*a_3|.*b_3|.*a_4|.*b_4"
    # f1_scores format is: {exp_id: [{n_samples: n_samples, f1_score: f1_score}]}
    key_n_samples = "data.pretraining.train_dataloader.dataset.n_samples"
    f1_scores = parse_result_folder(
        result_folder,
        dataset,
        run_filter,
        [key_n_samples],
    )

    # merge a_3 and a_4, b_3 and b_4
    f1_scores["exp007a_3"].extend(f1_scores["exp007a_4"])
    f1_scores["exp007b_3"].extend(f1_scores["exp007b_4"])
    del f1_scores["exp007a_4"]
    del f1_scores["exp007b_4"]
    print(f1_scores)

    # load additional f1 scores for fully supervised and untrained FE
    key_n_subjects = "data.downstream.train_dataloader.dataset.data_reducer.n_subjects"
    add_f1_scores = parse_result_folder(
        "../../logs/exp001",
        dataset,
        ".*a_0|.*d_0",
        [key_n_subjects],
    )
    print(add_f1_scores)
    # only keep the f1 scores for the number of subjects that we are interested in
    fully_supervised_scores = [
        e["f1_score"]
        for e in add_f1_scores["exp001a_0"]
        if e[key_n_subjects] == n_subjects
    ]
    untrained_fe_scores = [
        e["f1_score"]
        for e in add_f1_scores["exp001d_0"]
        if e[key_n_subjects] == n_subjects
    ]

    # all values of n_samples that are present in the results (assuming that all experiments have the same values)
    n_samples = sorted(set([e[key_n_samples] for e in f1_scores["exp007a_3"]]))

    my_cmap = plt.get_cmap("gist_earth")
    bar_width = 0.4

    legend_labels = []

    # add two bars for the fully supervised and untrained FE
    ax.bar(
        [-1],
        [np.mean(untrained_fe_scores)],
        yerr=[np.std(untrained_fe_scores)],
        color="red",
        width=bar_width,
        label="Untrained FE",
        hatch=patterns[3],
        alpha=1.0,
        capsize=3,
    )
    ax.bar(
        [-1 + bar_width],
        [np.mean(fully_supervised_scores)],
        yerr=[np.std(fully_supervised_scores)],
        color="black",
        width=bar_width,
        label="Fully supervised",
        hatch=patterns[2],
        alpha=1.0,
        capsize=3,
    )

    # add bars for the different number of synthetic samples
    for i, (exp, scores) in enumerate(f1_scores.items()):
        means = [
            np.mean([e["f1_score"] for e in scores if e[key_n_samples] == n])
            for n in n_samples
        ]
        stds = [
            np.std([e["f1_score"] for e in scores if e[key_n_samples] == n])
            for n in n_samples
        ]

        color = my_cmap((i + 1) / (len(f1_scores) + 2))
        legend_labels.append(
            [v for k, v in expid_to_label.items() if re.fullmatch(k, exp)][0]
        )

        ax.bar(
            np.arange(len(means)) + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            color=color,
            label=legend_labels[-1],
            hatch=patterns[i],
            alpha=1.0,
            capsize=3,
        )

    ax.set_xlabel("Number of synthetic samples")
    ax.set_ylabel("Macro F1 score")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(which="major", width=1.00, length=4)
    ax.set_xticks(
        [-1 + bar_width / 2] + [i + bar_width / 2 for i in range(len(n_samples))]
    )
    ax.set_xticklabels(["0"] + [f"{n:,}" for n in n_samples], rotation=30)
    ax.legend(loc="lower right")
    ax.grid(axis="both", linestyle="--", alpha=0.7)
    ax.set_ylim([0.2, 0.7])


def main():
    # relative path to the parent folder of the experiment folders that contain the results to be visualized in the
    # first 3 panels
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
