"""
This script creates Figure 2 in the paper. It visualizes the macro F1 scores on the test data for different datasets and
training regimes (low-data and high-data).
"""
import re
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils import parse_result_folder

sns.set_style("white")

patterns = ["//", "--", "xx", "oo"]

# mapping from experiment id to label in the plot
expid_to_label = {
    "exp0..a.*": "Fully supervised",
    "exp0..b.*": "Fixed feature extractor",
    "exp0..c.*": "Fine-tuned feature extractor",
    "exp0..d.*": "Untrained feature extractor",
}

# how to sort the experiments in the plot/legend
expid_to_sortkey = {
    "exp0..a.*": 3,
    "exp0..b.*": 1,
    "exp0..c.*": 2,
    "exp0..d.*": 0,
}

low_data_expid_to_dataset = {
    "exp004.*": "DODO/H",
    "exp005.*": "Sleep-EDFx",
    "exp006.*": "ISRUC",
}

high_data_expid_to_dataset = {
    "exp001.*": "DODO/H",
    "exp002.*": "Sleep-EDFx",
    "exp003.*": "ISRUC",
}

dataset_to_sortkey = {
    "DODO/H": 0,
    "Sleep-EDFx": 1,
    "ISRUC": 2,
}


def main():
    datasplit = "test"
    font_size = 10
    high_data_offset = 3
    extra_space = 0.4
    n_subj_low_data = 1
    n_epochs_low_data = 50
    n_subj_high_data = -1
    n_epochs_high_data = None

    # f1_scores format is: {exp_id: [{n_subjects: n_subjects, n_epochs: n_epochs, f1_score: f1_score}, ...]}
    n_subjects_key = "data.downstream.train_dataloader.dataset.data_reducer.n_subjects"
    n_epochs_key = "data.downstream.train_dataloader.dataset.data_reducer.n_epochs"

    # high data regime
    result_folder = join("../../logs/exp001", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    # exp001 has a special case where the max number of subjects was specified as 56 instead of -1, so we remap it here
    f1_scores = {
        exp: [
            {
                n_subjects_key: n_subj_high_data,
                n_epochs_key: e[n_epochs_key],
                "f1_score": e["f1_score"],
            }
            for e in f1_scores[exp]
            if e[n_subjects_key] == 56
        ]
        for exp in f1_scores.keys()
    }

    result_folder = join("../../logs/exp002", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    add_f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    f1_scores.update(add_f1_scores)

    result_folder = join("../../logs/exp003", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    add_f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    f1_scores.update(add_f1_scores)

    # low data regime
    result_folder = join("../../logs/exp004", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    add_f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    f1_scores.update(add_f1_scores)

    result_folder = join("../../logs/exp005", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    add_f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    f1_scores.update(add_f1_scores)

    result_folder = join("../../logs/exp006", ".")
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    add_f1_scores = parse_result_folder(
        result_folder, datasplit, run_filter, [n_subjects_key, n_epochs_key]
    )
    f1_scores.update(add_f1_scores)

    print(f1_scores)

    # map experiment ids to datasets
    datasets = set()
    for exp in f1_scores.keys():
        # first check in low data regime
        dataset = [
            v for k, v in low_data_expid_to_dataset.items() if re.fullmatch(k, exp)
        ]
        # then check in high data regime
        if len(dataset) == 0:
            dataset = [
                v for k, v in high_data_expid_to_dataset.items() if re.fullmatch(k, exp)
            ]
        if len(dataset) == 0:
            print(f"WARNING: no dataset found for {exp}")
            continue
        datasets.add(dataset[0])
    datasets = sorted(list(datasets), key=lambda x: dataset_to_sortkey[x])

    plt.rcParams.update({"font.size": font_size})
    plt.figure(figsize=(8.5, 4))
    my_cmap = plt.get_cmap("gist_earth")
    legend_labels = []

    for i, dataset in enumerate(datasets):
        # load scores of current dataset in low data regime
        f1_scores_low_data = {}
        for e in f1_scores.keys():
            e_dataset_matches = [
                v for k, v in low_data_expid_to_dataset.items() if re.fullmatch(k, e)
            ]
            # search for scores that match the required low data definition
            if len(e_dataset_matches) > 0 and e_dataset_matches[0] == dataset:
                f1_scores_low_data[e] = [
                    entry["f1_score"]
                    for entry in f1_scores[e]
                    if entry[n_subjects_key] == n_subj_low_data
                    and entry[n_epochs_key] == n_epochs_low_data
                ]

        exps_sorted = sorted(
            f1_scores_low_data.keys(),
            key=lambda x: [
                v for k, v in expid_to_sortkey.items() if re.fullmatch(k, x)
            ][0],
        )
        for j, exp in enumerate(exps_sorted):
            color = my_cmap((j + 1) / (len(exps_sorted) + 1))
            label = [v for k, v in expid_to_label.items() if re.fullmatch(k, exp)][0]

            mean = np.mean(f1_scores_low_data[exp])
            std = np.std(f1_scores_low_data[exp])
            rects = plt.bar(
                i + j * 0.2,
                mean,
                yerr=std,
                width=0.2,
                color=color,
                label=label if label not in legend_labels else None,
                hatch=patterns[j],
                alpha=1.0,
                capsize=3,
            )
            plt.bar_label(rects, labels=[f"{mean:.2f}"], padding=3, fontsize=font_size)
            legend_labels.append(label)

        # load scores of current dataset in high data regime
        f1_scores_high_data = {}
        for e in f1_scores.keys():
            e_dataset_matches = [
                v for k, v in high_data_expid_to_dataset.items() if re.fullmatch(k, e)
            ]
            # search for scores that match the required high data definition
            if len(e_dataset_matches) > 0 and e_dataset_matches[0] == dataset:
                f1_scores_high_data[e] = [
                    entry["f1_score"]
                    for entry in f1_scores[e]
                    if entry[n_subjects_key] == n_subj_high_data
                    and entry[n_epochs_key] == n_epochs_high_data
                ]

        exps_sorted = sorted(
            f1_scores_high_data.keys(),
            key=lambda x: [
                v for k, v in expid_to_sortkey.items() if re.fullmatch(k, x)
            ][0],
        )
        for j, exp in enumerate(exps_sorted):
            color = my_cmap((j + 1) / (len(exps_sorted) + 1))
            label = [v for k, v in expid_to_label.items() if re.fullmatch(k, exp)][0]

            mean = np.mean(f1_scores_high_data[exp])
            std = np.std(f1_scores_high_data[exp])
            rects = plt.bar(
                high_data_offset + extra_space + i + j * 0.2,
                mean,
                yerr=std,
                width=0.2,
                color=color,
                label=label if label not in legend_labels else None,
                hatch=patterns[j],
                alpha=1.0,
                capsize=3,
            )
            plt.bar_label(rects, labels=[f"{mean:.2f}"], padding=3, fontsize=font_size)
            legend_labels.append(label)

    plt.text(1.3, 0.93, "Low-data regime", ha="center", va="center", weight="bold")
    plt.text(
        4.3 + extra_space,
        0.93,
        "High-data regime",
        ha="center",
        va="center",
        weight="bold",
    )

    plt.grid(axis="y")
    plt.ylabel("Macro F1 score")
    plt.legend()

    plt.xticks(
        np.r_[
            np.arange(len(datasets)) + 0.3,
            np.arange(len(datasets)) + high_data_offset + extra_space + 0.3,
        ],
        datasets * 2,
    )
    plt.axvline(high_data_offset + extra_space / 2 - 0.2, color="black", linestyle="--")
    plt.ylim([0.2, 0.9])
    plt.legend(loc="upper left", ncol=1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
