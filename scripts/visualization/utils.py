import json
import re
from glob import glob
from os.path import isdir, join

import numpy as np


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
    result_folder: str,
    dataset: str,
    reg_filter: str,
    keys_to_read: list[str],
) -> dict[str, list[dict[str, any]]]:
    """
    Parse the parent folder of the separate experiment folders (e.g. exp002) and extract the F1 scores for the
    downstream task on the specified dataset. The F1 scores are grouped by the exp_id and paired with the values of the
    keys specified in keys_to_read. The keys_to_read should be a list of strings that correspond to the keys in the
    manual overrides of the training logs.

    :param result_folder: The parent folder of the experiment folders (e.g. path to exp001 in the logs)
    :param dataset: The dataset for which the F1 scores should be extracted. This should be the name of the dataset as
    it appears in the results.json file, e.g. "earlystopping" or "test".
    :param reg_filter: A regular expression to filter the experiment ids. Only experiment ids that match the regular
    expression will be included in the result. This functions expects the regex to filter for the sweeps that were used
    to evaluate the models by the experiment ids and the sweep index. E.g. ".*a_1|.*b_2" will include the second sweep
    of experiment a and the third sweep of experiment b (sweep indices are 0-based).
    :param keys_to_read: A list of strings that correspond to the keys in the manual overrides of the training logs that
    should be included in the result.

    :return: A dictionary containing the F1 scores for each experiment id and sweep index. The F1 scores are paired with
    the values of the keys specified in keys_to_read.
    Format: {exp_id: [{key1: val1, key2: val2, ..., "f1_score": f1_score}, ...]}
    """
    reg_filter = re.compile(reg_filter)
    # load the folder structure of a set of sub-experiments
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
    f1_scores = {}  # exp_id: [{key1: val1, key2: val2, ..., "f1_score": f1_score}, ...]

    for exp_f in folder_structure.keys():
        for i, sweep_folder in enumerate(folder_structure[exp_f].keys()):
            print(sweep_folder)
            # ignore folders that do not match the filter
            exp_id = f'{exp_f.split("/")[-1]}_{i}'
            if reg_filter.match(exp_id) is None:
                continue

            for folder in folder_structure[exp_f][sweep_folder]:
                if dataset == "earlystopping":
                    train_log_file = glob(join(folder, "*fine-tune.log"))
                else:
                    # if we want to parse the test scores, we first need to find the corresponding train log file
                    eval_log_file = glob(join(folder, "eval_fine-tuned.log"))[0]

                    # find matching train log file, i.e. the train log file that contains the path to the model that was
                    # evaluated
                    eval_run_overrides = read_log(eval_log_file)
                    # convert the string representation of the m_seed_path_sids variable to a dictionary by converting
                    # it to a valid json string and then parsing it
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

                    # search through all train log files in the folder structure of the experiment to find the one that
                    # contains the model path
                    all_train_log_files = np.concatenate(
                        [
                            glob(join(f, "*fine-tune.log"))
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

                # find the epoch with the highest f1 score on the validation set and extract the f1 score of this epoch
                # from the results file; in case of test evaluations, there is only one f1 score in the results file
                result_file = join(folder, "results.json")
                with open(result_file) as f:
                    results = json.load(f)
                # returns 0 if the list is empty (test evaluations)
                best_epoch = np.argmax(
                    results["earlystopping"]["metrics"]["downstream"]["f1_scores"][
                        "macro"
                    ]
                )
                run_f1_score = results[dataset]["metrics"]["downstream"]["f1_scores"][
                    "macro"
                ][best_epoch]

                # load all manual overrides from the train log file
                run_overrides = read_log(train_log_file[0])
                key_value_map = {k: None for k in keys_to_read}
                for param, val in run_overrides.items():
                    if param in keys_to_read:
                        key_value_map[param] = val
                if exp_id not in f1_scores:
                    f1_scores[exp_id] = []
                f1_scores[exp_id].append(key_value_map | {"f1_score": run_f1_score})

    return f1_scores
