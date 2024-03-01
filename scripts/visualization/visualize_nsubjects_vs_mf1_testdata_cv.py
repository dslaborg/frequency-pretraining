"""
This script creates Figure 2 in the paper. It visualizes the macro F1 scores on the test data for different numbers of
subjects in the training data. The data is from the experiment exp001, which is a comparison of different training
strategies for the downstream task.
"""
import json
import re
from glob import glob
from os.path import isdir, join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("white")

ignore_folders = [".hydra"]
ignore_keys = ["m_seed_path_sids"]
patterns = ["//", "--", "xx", "oo"]

# mapping from experiment id to label in the plot
expid_to_label = {
    "exp001a_1": "Fully supervised",
    "exp001b_2": "Fixed feature extractor",
    "exp001c_1": "Fine-tuned feature extractor",
    "exp001d_1": "Untrained feature extractor",
}

# how to sort the experiments in the plot/legend
expid_to_sortkey = {
    "exp001a_1": 3,
    "exp001b_2": 1,
    "exp001c_1": 2,
    "exp001d_1": 0,
}


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
) -> dict[str, dict[int, list[float]]]:
    """
    Parse the parent folder of the separate experiment folders (e.g. exp001) and extract the F1 scores for the
    downstream task on the test data. The F1 scores are grouped by the number of subjects in the training data and the
    experiment id.
    Since the training logs don't contain the test F1 scores and the test logs don't contain the number of subjects, we
    need to match the training and test logs by the model path. This is done by reading the "m_seed_path_sids" override
    from the test log and comparing it to the "m_seed_path_sids" override from the training log.

    :param result_folder: The parent folder of the experiment folders (e.g. path to exp001 in the logs)
    :param reg_filter: A regular expression to filter the experiment ids. Only experiment ids that match the regular
    expression will be included in the result. This functions expects the regex to filter for the sweeps that were used
    to evaluate the models by the experiment ids and the sweep index. E.g. ".*a_1|.*b_2" will include the second sweep
    of experiment a and the third sweep of experiment b (sweep indices are 0-based).

    :return: A dictionary containing the F1 scores for each experiment and number of subjects in the training data.
    Format: {exp_id: {n_subjects: [f1_scores]}}
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
    f1_scores = {}  # exp_id: n_subjects: [f1_score]

    for exp_f in folder_structure.keys():
        for i, sweep_folder in enumerate(folder_structure[exp_f].keys()):
            print(sweep_folder)
            sweep_f1_scores = {}  # n_subjects: [f1_score]
            exp_id = f'{exp_f.split("/")[-1]}_{i}'

            if reg_filter.match(exp_id) is None:
                continue

            for run_folder in folder_structure[exp_f][sweep_folder]:
                # find eval log file in run_folder
                eval_log_file = glob(join(run_folder, "*downstream*.log"))
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
                        glob(join(f, "train*downstream*.log"))
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
                for param, val in run_overrides.items():
                    if param in ignore_keys:
                        continue
                    elif (
                        param
                        == "data.downstream.train_dataloader.dataset.data_reducer.n_subjects"
                    ):
                        n_subjects = int(val)
                    else:
                        print(f"WARNING: unknown override key: {param}")

                if n_subjects in sweep_f1_scores:
                    sweep_f1_scores[n_subjects].append(run_f1_score)
                else:
                    sweep_f1_scores[n_subjects] = [run_f1_score]
            if len(sweep_f1_scores) != 0:
                f1_scores[exp_id] = sweep_f1_scores

    return f1_scores


def main():
    # relative path to the parent folder of the experiment folders that contain the results to be visualized
    base_path_of_file = "../../logs/exp001"
    result_folder = join(base_path_of_file, ".")
    # we want the 2nd sweep of experiment a, the 3rd sweep of experiment b, the 2nd sweep of experiment c and the 2nd
    # sweep of experiment d
    run_filter = ".*a_1|.*b_2|.*c_1|.*d_1"
    f1_scores = parse_result_folder(result_folder, run_filter)
    # f1_scores format is: {exp_id: {n_subjects: [f1_scores]}}
    print(f1_scores)

    plt.figure(figsize=(9, 6))
    my_cmap = plt.get_cmap("gist_earth")
    # my_cmap = plt.get_cmap('viridis')
    legend_labels = []
    # sort experiments by the sort key
    exps_sorted = sorted(f1_scores.keys(), key=lambda x: expid_to_sortkey[x])
    # plot the bars for each experiment
    for j, exp in enumerate(exps_sorted):
        color = my_cmap((j + 1) / (len(exps_sorted) + 1))
        legend_labels.append(expid_to_label[exp])

        # sort the number of subjects in ascending order
        means = np.array([np.mean(v) for v in f1_scores[exp].values()])[::-1]
        stds = np.array([np.std(v) for v in f1_scores[exp].values()])[::-1]
        rects = plt.bar(
            np.arange(len(means)) + j * 0.2,
            means,
            yerr=stds,
            width=0.2,
            color=color,
            label=expid_to_label[exp],
            hatch=patterns[j],
            alpha=1.0,
            capsize=3,
        )
        plt.bar_label(rects, labels=[f"{m:.2f}" for m in means], padding=3, fontsize=9)

    plt.grid()
    plt.xlabel("Number of subjects")
    plt.ylabel("Macro F1 score")
    plt.legend()

    n_subjects_all = list(list(f1_scores.values())[0].keys())[::-1]
    plt.xticks(np.arange(len(n_subjects_all)) + 0.3, [f"{n:g}" for n in n_subjects_all])
    plt.ylim([0.25, 0.86])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
