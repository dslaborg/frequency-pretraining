import json
import re
from glob import glob
from os.path import isdir, join

import numpy as np

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
    "general.gpus",
]


def read_log(path):
    with open(path) as f:
        all_lines = [line.removesuffix("\n") for line in f.readlines()]

        override_start_line = [
            i for i, line in enumerate(all_lines) if line == "task:"
        ][0]
        overrides = {}
        for line in all_lines[override_start_line + 1 :]:
            if line.startswith("-"):
                override_line = line[2:].split("=")
                overrides[override_line[0]] = override_line[1]
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


def parse_result_folder(result_folder, dataset, reg_filter):
    reg_filter = re.compile(reg_filter)
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
    f1_scores = {}  # exp_id: n_subjects: n_epochs: f1_scores

    for exp_f in folder_structure.keys():
        for i, sweep_folder in enumerate(folder_structure[exp_f].keys()):
            print(sweep_folder)
            sweep_f1_scores = {}  # n_subjects: n_epochs: f1_scores
            exp_id = f'{exp_f.split("/")[-1]}_{i}'
            if reg_filter.match(exp_id) is None:
                continue
            comment = None
            for folder in folder_structure[exp_f][sweep_folder]:
                if dataset == "earlystopping":
                    train_log_file = glob(join(folder, "fine_tune.log"))
                else:
                    eval_log_file = glob(join(folder, "eval_fine-tuned.log"))
                    if len(eval_log_file) == 0:
                        print(f"WARNING: no downstream log file found in {folder}")
                        continue
                    eval_log_file = eval_log_file[0]

                    # find matching train log file
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
                    # fold = path_and_fold["subject_ids"]["dod_o_h"]
                    # print(fold)
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

                run_overrides = read_log(train_log_file[0])

                result_file = join(folder, "results.json")
                with open(result_file) as f:
                    results = json.load(f)
                best_epoch = np.argmax(
                    results["earlystopping"]["metrics"]["downstream"]["f1_scores"][
                        "macro"
                    ]
                )
                run_f1_score = results[dataset]["metrics"]["downstream"]["f1_scores"][
                    "macro"
                ][best_epoch]

                n_epochs = -1
                n_subjects = 0
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
                        n_subjects = n_subjects if n_subjects != -1 else 999
                    elif param == "+comment":
                        comment = val.replace("\\", "")
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
                f1_scores[exp_id + (f" - {comment}" if comment is not None else "")] = (
                    sweep_f1_scores
                )

    return f1_scores


if __name__ == "__main__":
    datasplit = "test"
    exps = ["exp008", "exp009", "exp010", "exp011"]
    for exp in exps:
        print(f"Parsing {exp}...")
        result_folder = join("../../../logs", exp, ".")
        run_filter = ".*a_3|.*b_2|.*a_4|.*b_3"
        # f1_scores format is: {exp_id: {n_subjects: {n_epochs: [f1_scores]}}}
        f1_scores = parse_result_folder(result_folder, datasplit, run_filter)
        # merge runs of the same experiment
        merged_f1_scores = {}
        for exp_id, sweep_f1_scores in f1_scores.items():
            exp_id_short = exp_id[:-2]  # remove comment
            if exp_id_short not in merged_f1_scores:
                merged_f1_scores[exp_id_short] = {}
            for n_subjects, n_epochs_f1_scores in sweep_f1_scores.items():
                if n_subjects not in merged_f1_scores[exp_id_short]:
                    merged_f1_scores[exp_id_short][n_subjects] = {}
                for n_epochs, f1_scores in n_epochs_f1_scores.items():
                    if n_epochs not in merged_f1_scores[exp_id_short][n_subjects]:
                        merged_f1_scores[exp_id_short][n_subjects][n_epochs] = []
                    merged_f1_scores[exp_id_short][n_subjects][n_epochs].extend(
                        f1_scores
                    )
        print(merged_f1_scores)
