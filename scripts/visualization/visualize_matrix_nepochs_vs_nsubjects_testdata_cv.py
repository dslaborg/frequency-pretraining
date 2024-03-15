"""
This script creates Figure 3 in the paper. It visualizes the macro F1 scores on the test data for different numbers of
subjects and numbers of epochs used for fine-tuning. The figure consists of three subplots: a) the mean F1 scores for
exp002a, b) the mean F1 scores for exp002b, and c) the difference between the mean F1 scores for exp002b and exp002a.
The difference is calculated as exp002b - exp002a with a bootstrapping approach , i.e. positive values indicate that
exp002b performed better than exp002a.
"""
import json
import re
from glob import glob
from os.path import isdir, join, dirname, abspath

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("white")

ignore_folders = [".hydra"]
ignore_keys = [
    "+dummy",
    "+ignore",
    "m_seed_path",
    "seeds",
    "data.dod_o_h.channels",
    "data.normalize",
    "data.norm_length",
    "m_seed_path_sids",
]
colors = ["red", "green", "blue", "orange", "purple", "cyan", "black", "grey"]


def read_log(path: str) -> dict[str, any]:
    """
    Read the log file of a training/evaluation run and extract the overrides that were used for the run. The overrides
    contain parameters set specifically for this run, e.g. the number of subjects in the training data or the seeds.
    """
    with open(path) as f:
        all_lines = [line.removesuffix("\n") for line in f.readlines()]

        # the override block in the logs starts with "task:" and ends with the next line that does not start with a
        # "-" character
        override_start_line = [
            i for i, line in enumerate(all_lines) if line == "task:"
        ][0]
        overrides = {}
        for line in all_lines[override_start_line + 1 :]:
            if line.startswith("-"):
                # format of the overrides is "- key=value"
                override_line = line[2:].split("=")
                overrides[override_line[0]] = override_line[1]
                # try to convert the value to int or float if possible
                try:
                    overrides[override_line[0]] = int(override_line[1])
                except ValueError:
                    pass
                try:
                    overrides[override_line[0]] = float(override_line[1])
                except ValueError:
                    pass
                if override_line[1].startswith('"') and override_line[1].endswith('"'):
                    overrides[override_line[0]] = override_line[1][1:-1]
            else:
                break
    return overrides


def parse_result_folder(
    result_folder: str, reg_filter: str
) -> dict[str, dict[int, dict[int, list[float]]]]:
    """
    Parse the parent folder of the separate experiment folders (e.g. exp002) and extract the F1 scores for the
    downstream task on the test data. The F1 scores are grouped by the number of subjects and epochs in the training
    data and the experiment id.
    Since the training logs don't contain the test F1 scores and the test logs don't contain the number of subjects, we
    need to match the training and test logs by the model path. This is done by reading the "m_seed_path_sids" override
    from the test log and comparing it to the "m_seed_path_sids" override from the training log.

    :param result_folder: The parent folder of the experiment folders (e.g. path to exp001 in the logs)
    :param reg_filter: A regular expression to filter the experiment ids. Only experiment ids that match the regular
    expression will be included in the result. This functions expects the regex to filter for the sweeps that were used
    to evaluate the models by the experiment ids and the sweep index. E.g. ".*a_1|.*b_2" will include the second sweep
    of experiment a and the third sweep of experiment b (sweep indices are 0-based).

    :return: A dictionary containing the F1 scores for each experiment, number of subjects, and number of epochs in the
    training data.
    Format: {exp_id: {n_subjects: {n_epochs: [f1_scores]}}}
    """

    reg_filter = re.compile(reg_filter)
    # folder_structure is a nested dictionary that contains the folder structure of the experiment folders. It is
    # structured as follows: {exp_folder: {sweep_folder: [run_folders]}}
    folder_structure = {
        ef: {
            sw: sorted(
                [p for p in glob(join(sw, "*")) if isdir(p)],
                key=lambda x: int(x.split("/")[-1]),
            )
            for sw in sorted(glob(f"{ef}/sweep*"))
        }
        for ef in sorted(glob(f"{result_folder}/*"))
    }
    f1_scores = {}  # exp_id: n_subjects: n_epochs: [f1_score]

    for exp_f in folder_structure.keys():
        for i, sweep_folder in enumerate(folder_structure[exp_f].keys()):
            print(sweep_folder)
            sweep_f1_scores = {}  # n_subjects: n_epochs: [f1_score]
            exp_id = f'{exp_f.split("/")[-1]}_{i}'

            if reg_filter.match(exp_id) is None:
                continue

            for run_folder in folder_structure[exp_f][sweep_folder]:
                # find eval log file in run_folder
                eval_log_file = glob(join(run_folder, "*.log"))
                if len(eval_log_file) == 0:
                    print(f"WARNING: no downstream log file found in {run_folder}")
                    continue
                eval_log_file = eval_log_file[0]

                # find matching train log file by searching for the train log that contains the model path from the eval
                # log
                # read eval log and extract model path
                eval_run_overrides = read_log(eval_log_file)
                m_seed_path_sids = re.sub(
                    ':([^{"].*?[^\\\])([,}])',
                    ':"\g<1>"\g<2>',
                    eval_run_overrides["m_seed_path_sids"],
                )
                m_seed_path_sids = re.sub(
                    "([{,])(.*?):", '\g<1>"\g<2>":', m_seed_path_sids
                )
                m_seed_path_sids = m_seed_path_sids.replace("\\", "\\\\")
                path_and_fold = json.loads(m_seed_path_sids)
                model_path = path_and_fold["path"]

                # load all train log files and find the one that contains the model path
                all_train_log_files = np.concatenate(
                    [
                        glob(join(f, "fine-tune.log"))
                        for sw in folder_structure[exp_f].keys()
                        for f in folder_structure[exp_f][sw]
                    ]
                )
                train_log_file = list(
                    filter(
                        lambda x: model_path in "".join(open(x).readlines()),
                        all_train_log_files,
                    )
                )
                if len(train_log_file) == 0 or len(train_log_file) > 1:
                    print(
                        f"WARNING: no matching train log file or multiple matching files found for {model_path}"
                    )
                    continue

                # read eval log and extract F1 score
                eval_result_file = join(run_folder, "results.json")
                with open(eval_result_file) as f:
                    results = json.load(f)
                run_f1_score = results["test"]["metrics"]["downstream"]["f1_scores"][
                    "macro"
                ][0]

                # read train log and extract number of subjects
                run_overrides = read_log(train_log_file[0])
                n_subjects = 0
                n_epochs = 0
                for param, val in run_overrides.items():
                    if param in ignore_keys:
                        continue
                    if (
                        param
                        == "data.downstream.train_dataloader.dataset.data_reducer.n_epochs"
                    ):
                        n_epochs = int(val)
                    elif (
                        param
                        == "data.downstream.train_dataloader.dataset.data_reducer.n_subjects"
                    ):
                        n_subjects = int(val)
                    else:
                        print(f"WARNING: unknown override key: {param}")

                if n_subjects in sweep_f1_scores:
                    if n_epochs in sweep_f1_scores[n_subjects]:
                        sweep_f1_scores[n_subjects][n_epochs].append(run_f1_score)
                    else:
                        sweep_f1_scores[n_subjects][n_epochs] = [run_f1_score]
                else:
                    sweep_f1_scores[n_subjects] = {n_epochs: [run_f1_score]}
            if len(sweep_f1_scores) != 0:
                f1_scores[exp_id] = sweep_f1_scores

    return f1_scores


def plot_matrix(
    matrix_color_values: np.ndarray,
    matrix_print_values: np.ndarray,
    axis: plt.Axes,
    title: str,
    xticks: list[str],
    yticks: list[str],
    max_color: float = 1.0,
    min_color: float = 0.0,
):
    """
    Plot a matrix with the given color values and print values. The color values are used to color the cells of the
    matrix, the print values are used as labels inside the cells.

    :param matrix_color_values: The values that are used to color the cells of the matrix.
    :param matrix_print_values: The values that are printed inside the cells of the matrix.
    :param axis: The axis on which the matrix is plotted.
    :param title: The title of the plot.
    :param xticks: The labels for the x-axis.
    :param yticks: The labels for the y-axis.
    :param max_color: The maximum value of the color scale. Allows to set the same color scale for multiple plots even
    though the maximum value in matrix_color_values might differ.
    :param min_color: The minimum value of the color scale. Allows to set the same color scale for multiple plots even
    though the minimum value in matrix_color_values might differ.
    :return:
    """
    assert matrix_color_values.shape == matrix_print_values.shape, (
        f"The shape of the color matrix {matrix_color_values.shape} does not match the shape of the "
        f"label matrix {matrix_print_values.shape}"
    )
    # plot the colored regions of the matrix
    axis.imshow(
        matrix_color_values,
        interpolation="nearest",
        cmap="Blues",
        vmin=min_color,
        vmax=max_color,
    )
    # we want to show all ticks...
    axis.set(
        xticks=np.arange(matrix_print_values.shape[1]),
        yticks=np.arange(matrix_print_values.shape[0]),
        # ... and label them with the respective tick labels.
        xticklabels=xticks,
        yticklabels=yticks,
        title=title,
    )
    plt.ylim(matrix_print_values.shape[0] - 0.5, -0.5)

    # loop over data dimensions and create text annotations.
    thresh = (max_color + min_color) / 2  # threshold for text color (black or white)
    for i in range(matrix_print_values.shape[0]):
        for j in range(matrix_print_values.shape[1]):
            axis.text(
                j,
                i,
                str(matrix_print_values[i, j]),
                ha="center",
                va="center",
                color="white" if matrix_color_values[i, j] > thresh else "black",
            )


def main():
    # relative path to the experiment folder containing the sub-experiments
    base_path_of_file = dirname(abspath(__file__))
    result_folder = join(base_path_of_file, "../../logs/exp002")

    # we want the results of the second multirun of exp002a and the third multirun of exp002b
    experiment1 = "exp002a_1"
    experiment2 = "exp002b_2"
    # f1_scores format is: {exp_id: {n_subjects: {n_epochs: [f1_scores]}}}
    f1_scores = parse_result_folder(result_folder, f".*{experiment1}|.*{experiment2}")
    print(f1_scores)

    # extract the number of subjects and epochs from the f1_scores dictionary
    # the -1 value should be at the end instead of the start of the list, so we remove the first element and append -1
    n_epochs_list = list(list(list(f1_scores.values())[0].values())[0])[1:] + [-1]
    # reverse the list for the correct order in the plot
    n_epochs_list = n_epochs_list[::-1]
    n_subjects_list = list(f1_scores.values())[0].keys()

    # plot a matrix of n_epochs vs n_subjects showing the difference between the mf1 score between exp002a and exp002b
    # values for panel a: the labels are string representations of the mean f1 scores with standard deviation
    # the color values are the mean f1 scores
    matrix_labels_panel_a = [
        ["" for _ in range(len(n_subjects_list))] for _ in range(len(n_epochs_list))
    ]
    matrix_color_panel_a = np.zeros((len(n_epochs_list), len(n_subjects_list)))

    # values for panel b: the labels are string representations of the mean f1 scores with standard deviation
    # the color values are the mean f1 scores
    matrix_labels_panel_b = [
        ["" for _ in range(len(n_subjects_list))] for _ in range(len(n_epochs_list))
    ]
    matrix_color_panel_b = np.zeros((len(n_epochs_list), len(n_subjects_list)))

    # values for panel c: the labels are string representations of the difference between the mean f1 scores
    # (calculated as exp002b - exp002a with a bootstrapping approach) with standard deviations of the bootstrapped
    # differences
    # the color values are the mean differences between the bootstrapped mean f1 scores
    matrix_color_panel_c = np.zeros((len(n_epochs_list), len(n_subjects_list)))
    matrix_labels_panel_c = [
        ["" for _ in range(len(n_subjects_list))] for _ in range(len(n_epochs_list))
    ]

    for i, n_epochs in enumerate(n_epochs_list):
        for j, n_subjects in enumerate(n_subjects_list):
            exp1_scores = f1_scores[experiment1][n_subjects][n_epochs]
            exp2_scores = f1_scores[experiment2][n_subjects][n_epochs]

            # fill labels and colors for panel a and b
            matrix_labels_panel_a[i][
                j
            ] = f"{np.mean(exp1_scores):.2f}\n±{np.std(exp1_scores):.2f}"
            matrix_color_panel_a[i][j] = np.mean(exp1_scores)

            matrix_labels_panel_b[i][
                j
            ] = f"{np.mean(exp2_scores):.2f}\n±{np.std(exp2_scores):.2f}"
            matrix_color_panel_b[i][j] = np.mean(exp2_scores)

            # bootstrap the difference between the mean f1 scores
            # the sample size is equal to the number of runs
            sample_size = min(len(exp1_scores), len(exp2_scores))
            # calculate 10,000 bootstrap samples, each bootstrap sample is the difference between the mean f1 scores
            # of a random sample of scores from exp002b and exp002a
            bootstrap_samples = np.array(
                [
                    np.mean(np.random.choice(exp2_scores, sample_size))
                    - np.mean(np.random.choice(exp1_scores, sample_size))
                    for _ in range(10_000)
                ]
            )
            # calculate the mean and standard deviation of the bootstrap samples
            diff_mean = np.mean(bootstrap_samples)
            diff_std = np.std(bootstrap_samples)

            # fill labels and colors for panel c
            matrix_labels_panel_c[i][j] = f"{diff_mean:.2f}\n±{diff_std:.2f}"
            matrix_color_panel_c[i, j] = diff_mean

    # convert labels to numpy arrays for plotting
    matrix_labels_panel_a = np.array(matrix_labels_panel_a)
    matrix_labels_panel_b = np.array(matrix_labels_panel_b)
    matrix_labels_panel_c = np.array(matrix_labels_panel_c)

    # round color values to two decimal places equivalent to the labels
    matrix_color_panel_c = np.round(matrix_color_panel_c, 2)
    matrix_color_panel_a = np.round(matrix_color_panel_a, 2)
    matrix_color_panel_b = np.round(matrix_color_panel_b, 2)

    # plot with three panels
    fig, axes = plt.subplots(1, 3, sharex="all", sharey="all")
    axes = axes.flatten()
    # x-axis labels are the number of subjects, y-axis labels are the number of epochs
    x_ticks = [f"{n_subjects}" for n_subjects in n_subjects_list]
    y_ticks = [
        f"{n_epochs}" if n_epochs > 0 else "all" for n_epochs in n_epochs_list
    ]  # replace -1 with "all"

    # panels a and b share the same color scale, so we calculate the minimum and maximum color values for both panels
    min_color = np.min([np.min(matrix_color_panel_a), np.min(matrix_color_panel_b)])
    max_color = np.max([np.max(matrix_color_panel_a), np.max(matrix_color_panel_b)])
    plot_matrix(
        matrix_color_panel_a,
        matrix_labels_panel_a,
        axes[0],
        "",
        x_ticks,
        y_ticks,
        max_color,
        min_color,
    )
    axes[0].set_xlabel("Number of subjects")
    axes[0].set_ylabel("Number of epochs")
    plt.colorbar(axes[0].get_images()[0], ax=axes[0])

    plot_matrix(
        matrix_color_panel_b,
        matrix_labels_panel_b,
        axes[1],
        "",
        x_ticks,
        y_ticks,
        max_color,
        min_color,
    )
    axes[1].set_xlabel("Number of subjects")
    plt.colorbar(axes[1].get_images()[0], ax=axes[1])

    plot_matrix(
        matrix_color_panel_c,
        matrix_labels_panel_c,
        axes[2],
        "",
        x_ticks,
        y_ticks,
        np.max(matrix_color_panel_c),
        np.min(matrix_color_panel_c),
    )
    axes[2].set_xlabel("Number of subjects")
    plt.colorbar(axes[2].get_images()[0], ax=axes[2])

    plt.text(0.48, 1.03, "a", transform=axes[0].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "b", transform=axes[1].transAxes, size=12, weight="bold")
    plt.text(0.48, 1.03, "c", transform=axes[2].transAxes, size=12, weight="bold")

    plt.show()


if __name__ == "__main__":
    main()
