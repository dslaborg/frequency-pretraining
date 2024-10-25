"""
This script creates Figure 3 in the paper. It visualizes the macro F1 scores on the test data for different datasets
(DODO/H, Sleep-EDFx, ISRUC) and different numbers of subjects and epochs used for fine-tuning. The figure consists of
nine subplots: a) the mean F1 scores for exp004a, b) the mean F1 scores for exp005a, c) the mean F1 scores for exp006a,
d) the mean F1 scores for exp004c, e) the mean F1 scores for exp005c, f) the mean F1 scores for exp006c, g) the difference
between the mean F1 scores for exp004c and exp004a, h) the difference between the mean F1 scores for exp005c and exp005a,
i) the difference between the mean F1 scores for exp006c and exp006a.
The difference is calculated as exp004b - exp004a with a bootstrapping approach, i.e. positive values indicate that
exp004b performed better than exp004a.
"""

import re

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils import parse_result_folder

sns.set_style("white")

expid_to_sortkey = {
    "exp0..a.*": 0,
    "exp0..c.*": 1,
}

expid_to_dataset = {
    "exp004.*": "DODO/H",
    "exp005.*": "Sleep-EDFx",
    "exp006.*": "ISRUC",
}

dataset_to_sortkey = {
    "DODO/H": 0,
    "Sleep-EDFx": 1,
    "ISRUC": 2,
}

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def plot_matrix(
    matrix_colors,
    matrix_values,
    axis,
    xticks,
    yticks,
    max_color=1.0,
    min_color=0.0,
):
    assert matrix_colors.shape == matrix_values.shape
    axis.imshow(
        matrix_colors,
        interpolation="nearest",
        cmap="Blues",
        vmin=min_color,
        vmax=max_color,
    )
    # we want to show all ticks...
    axis.set(
        xticks=np.arange(matrix_values.shape[1]),
        yticks=np.arange(matrix_values.shape[0]),
        # ... and label them with the respective stages
        xticklabels=xticks,
        yticklabels=yticks,
    )
    plt.ylim(matrix_values.shape[0] - 0.5, -0.5)

    # rotate the tick labels and set their alignment.
    plt.setp(
        axis.get_xticklabels()
    )  # , rotation=45, ha='right', rotation_mode='anchor')

    # loop over data dimensions and create text annotations.
    thresh = (max_color + min_color) / 2
    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            axis.text(
                j,
                i,
                matrix_values[i, j],
                ha="center",
                va="center",
                color="white" if matrix_colors[i, j] > thresh else "black",
            )


def main():
    dataset = "test"
    error_mode = "bootstrap"  # 'simple', 'bootstrap', 'error_propagation'

    # f1_scores format is: {exp_id: [{n_subjects: n_subjects, n_epochs: n_epochs, f1_score: f1_score}, ...]}
    n_epochs_key = "data.downstream.train_dataloader.dataset.data_reducer.n_epochs"
    n_subjects_key = "data.downstream.train_dataloader.dataset.data_reducer.n_subjects"
    result_folder = "../../logs/exp004"
    f1_scores = parse_result_folder(
        result_folder, dataset, ".*a_1|.*c_1", [n_epochs_key, n_subjects_key]
    )

    result_folder = "../../logs/exp005"
    add_f1_scores = parse_result_folder(
        result_folder, dataset, ".*a_1|.*c_1", [n_epochs_key, n_subjects_key]
    )
    f1_scores.update(add_f1_scores)

    result_folder = "../../logs/exp006"
    add_f1_scores = parse_result_folder(
        result_folder, dataset, ".*a_1|.*c_1", [n_epochs_key, n_subjects_key]
    )
    f1_scores.update(add_f1_scores)

    print(f1_scores)

    # list of datasets represented by the experiments
    datasets = sorted(
        list(
            {
                [v for k, v in expid_to_dataset.items() if re.fullmatch(k, exp)][0]
                for exp in f1_scores.keys()
            }
        ),
        key=lambda x: dataset_to_sortkey[x],
    )

    _, axes = plt.subplots(
        3, len(datasets), sharex="all", sharey="all", figsize=(15, 15)
    )
    axes = axes.flatten()

    for col_idx, dataset in enumerate(datasets):
        # load scores of the current dataset
        f1_scores_dataset = {
            e: f1_scores[e]
            for e in f1_scores.keys()
            if [v for k, v in expid_to_dataset.items() if re.fullmatch(k, e)][0]
            == dataset
        }

        n_epochs_list = [
            e[n_epochs_key] for e_arr in f1_scores_dataset.values() for e in e_arr
        ]
        n_epochs_list = list(set(n_epochs_list))
        n_epochs_list = sorted(
            n_epochs_list, key=lambda x: 1e6 if x == -1 else x, reverse=True
        )
        n_subjects_list = [
            e[n_subjects_key] for e_arr in f1_scores_dataset.values() for e in e_arr
        ]
        n_subjects_list = sorted(list(set(n_subjects_list)))

        # matrices to store the mean f1 scores and their standard deviations of the two experiments and their difference
        # the last dimension of the matrices is used to store the mean and the standard deviation
        matrix_raw_diff = np.zeros((len(n_epochs_list), len(n_subjects_list), 2))
        matrix_raw_1 = np.zeros((len(n_epochs_list), len(n_subjects_list), 2))
        matrix_raw_2 = np.zeros((len(n_epochs_list), len(n_subjects_list), 2))

        exps_sorted = sorted(
            f1_scores_dataset.keys(),
            key=lambda x: [
                v for k, v in expid_to_sortkey.items() if re.fullmatch(k, x)
            ][0],
        )
        exp1_key, exp2_key = exps_sorted

        for i, n_epochs in enumerate(n_epochs_list):
            for j, n_subjects in enumerate(n_subjects_list):
                # load scores for n_epochs and n_subjects
                exp1_scores = [
                    e["f1_score"]
                    for e in f1_scores_dataset[exp1_key]
                    if e[n_epochs_key] == n_epochs and e[n_subjects_key] == n_subjects
                ]
                exp2_scores = [
                    e["f1_score"]
                    for e in f1_scores_dataset[exp2_key]
                    if e[n_epochs_key] == n_epochs and e[n_subjects_key] == n_subjects
                ]
                if error_mode == "simple":
                    diff_mean = np.mean(exp2_scores) - np.mean(exp1_scores)
                    diff_std = np.std(exp2_scores) - np.std(exp1_scores)
                elif error_mode == "bootstrap":
                    # create 10_000 bootstrap samples and calculate their mean and standard deviation
                    # bootstrap samples are defined as the difference between the mean f1 score of a random sample of
                    # exp2_scores and the mean f1 score of a random sample of exp1_scores
                    sample_size = min(len(exp1_scores), len(exp2_scores))
                    bootstrap_samples = np.array(
                        [
                            np.mean(np.random.choice(exp2_scores, sample_size))
                            - np.mean(np.random.choice(exp1_scores, sample_size))
                            for _ in range(10_000)
                        ]
                    )
                    diff_mean = np.mean(bootstrap_samples)
                    diff_std = np.std(bootstrap_samples)
                elif error_mode == "error_propagation":
                    # see https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
                    diff_mean = np.mean(exp2_scores) - np.mean(exp1_scores)
                    diff_std = np.sqrt(
                        np.std(exp2_scores) ** 2 + np.std(exp1_scores) ** 2
                    )
                else:
                    raise ValueError(f"Unknown error mode: {error_mode}")

                matrix_raw_diff[i, j, :] = diff_mean, diff_std
                matrix_raw_1[i, j, :] = np.mean(exp1_scores), np.std(exp1_scores)
                matrix_raw_2[i, j, :] = np.mean(exp2_scores), np.std(exp2_scores)

        matrix_raw_diff = np.round(matrix_raw_diff, 2)
        matrix_raw_1 = np.round(matrix_raw_1, 2)
        matrix_raw_2 = np.round(matrix_raw_2, 2)

        def pretty_print(matrix):
            """Pretty print a matrix with mean and standard deviation in the following format: mean\n±std"""
            return np.array(
                [
                    [
                        f"{matrix[i, j, 0]:.2f}\n±{matrix[i, j, 1]:.2f}"
                        for j in range(len(n_subjects_list))
                    ]
                    for i in range(len(n_epochs_list))
                ]
            )

        matrix_print_diff = pretty_print(matrix_raw_diff)
        matrix_print_1 = pretty_print(matrix_raw_1)
        matrix_print_2 = pretty_print(matrix_raw_2)

        y_ticks = [
            f"{n_epochs}" if n_epochs > 0 else "all" for n_epochs in n_epochs_list
        ]

        # use the same coloring for the scores of exp1 and exp2
        min_color = np.min(
            [np.min(matrix_raw_1[:, :, 0]), np.min(matrix_raw_2[:, :, 0])]
        )
        max_color = np.max(
            [np.max(matrix_raw_1[:, :, 0]), np.max(matrix_raw_2[:, :, 0])]
        )
        plot_matrix(
            matrix_raw_1[:, :, 0],
            matrix_print_1,
            axes[col_idx],
            n_subjects_list,
            y_ticks,
            max_color,
            min_color,
        )
        plt.colorbar(axes[col_idx].get_images()[0], ax=axes[col_idx])
        axes[col_idx].set_title(letters[col_idx], weight="bold")

        plot_matrix(
            matrix_raw_2[:, :, 0],
            matrix_print_2,
            axes[col_idx + len(datasets)],
            n_subjects_list,
            y_ticks,
            max_color,
            min_color,
        )
        plt.colorbar(
            axes[col_idx + len(datasets)].get_images()[0],
            ax=axes[col_idx + len(datasets)],
        )
        axes[col_idx + len(datasets)].set_title(
            letters[col_idx + len(datasets)], weight="bold"
        )

        plot_matrix(
            matrix_raw_diff[:, :, 0],
            matrix_print_diff,
            axes[col_idx + 2 * len(datasets)],
            n_subjects_list,
            y_ticks,
            np.max(matrix_raw_diff[:, :, 0]),
            np.min(matrix_raw_diff[:, :, 0]),
        )
        plt.colorbar(
            axes[col_idx + 2 * len(datasets)].get_images()[0],
            ax=axes[col_idx + 2 * len(datasets)],
        )
        axes[col_idx + 2 * len(datasets)].set_title(
            letters[col_idx + 2 * len(datasets)], weight="bold"
        )

        axes[col_idx + 2 * len(datasets)].set_xlabel("Number of subjects")
    axes[0].set_ylabel("Fully Supervised\nNumber of epochs")
    axes[len(datasets)].set_ylabel("Fine-tuned feature extractor\nNumber of epochs")
    axes[2 * len(datasets)].set_ylabel("Bootstrapped difference\nNumber of epochs")

    plt.text(
        0.5,
        1.08,
        "DODO/H",
        transform=axes[0].transAxes,
        size=12,
        weight="bold",
        horizontalalignment="center",
    )
    plt.text(
        0.5,
        1.08,
        "Sleep-EDFx",
        transform=axes[1].transAxes,
        size=12,
        weight="bold",
        horizontalalignment="center",
    )
    plt.text(
        0.5,
        1.08,
        "ISRUC",
        transform=axes[2].transAxes,
        size=12,
        weight="bold",
        horizontalalignment="center",
    )
    plt.text(
        -0.4,
        0.5,
        "Fully supervised",
        transform=axes[0].transAxes,
        size=12,
        weight="bold",
        rotation=90,
        verticalalignment="center",
    )
    plt.text(
        -0.4,
        0.5,
        "Fine-tuned feature extractor",
        transform=axes[3].transAxes,
        size=12,
        weight="bold",
        rotation=90,
        verticalalignment="center",
    )
    plt.text(
        -0.4,
        0.5,
        "Bootstrapped difference",
        transform=axes[6].transAxes,
        size=12,
        weight="bold",
        rotation=90,
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
