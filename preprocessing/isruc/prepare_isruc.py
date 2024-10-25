"""
This script prepares the ISRUC dataset for the dataloader. It reads the PSG files and the annotation files and saves the
signals and the hypnograms in numpy files. The signals are downsampled to 100 Hz and filtered with a low- and highpass
filter. The hypnograms are taken from the first annotator if possible, otherwise from the second annotator.
"""

import argparse
import os
import shutil
import sys
import tempfile
from glob import glob
from os.path import dirname, expanduser, join, realpath

import numpy as np
import pandas as pd
from mne.io import read_raw_edf

sys.path.insert(0, realpath(join(dirname(__file__), "../..")))

from preprocessing.downsample import downsample

# some channels have different names in the PSG files; all channels in a group re renamed to the same name
eeg_channels = [
    ("C3-A2", "C3-M2"),
    ("F3-A2", "F3-M2"),
    ("C4-A1", "C4-M1"),
    ("F4-A1", "F4-M1"),
    ("O1-A2", "O1-M2"),
    ("O2-A1", "O2-M1"),
]
emg_channels = []
eog_channels = [("LOC-A2", "E1-M2"), ("ROC-A1", "E2-M1")]
SAMPLING_RATE = 100
EPOCH_SEC_SIZE = 30
# combine N3 nd N4 stages into N3
STAGE_DICT = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3, "5": 4}


def extract_from_txt(file_path: str, stage_dict=STAGE_DICT) -> np.ndarray:
    """
    Extract hypnogram from txt file.
    """
    stages = np.genfromtxt(file_path, dtype=str)
    stages = np.array([stage_dict[s] for s in stages])

    return stages


def extract_from_rec(func):
    """
    Wrapper function to extract data from rec file by first converting it to edf.

    :param func: function to extract data from edf file
    :return: wrapper function
    """

    def extractor(file_path):
        # create tempfile from file_path with edf extension instead of rec extension
        temp_file_path = tempfile.mktemp(suffix=".edf")
        os.symlink(file_path, temp_file_path)
        # extract header from temp file
        data = func(temp_file_path)
        # delete temp file
        os.remove(temp_file_path)
        return data

    return extractor


def extract_from_edf(psg_file_path: str) -> tuple[np.ndarray, dict]:
    """
    Extracts data and metadata from an EDF (European Data Format) file.

    :param psg_file_path: The file path to the PSG (Polysomnography) EDF file.
    :return: A tuple containing:
        - data (np.ndarray): The extracted data from the EDF file, transposed to have shape (n_samples, n_channels).
        - meta_info (dict): A dictionary containing metadata with keys:
            - "sample_rate" (int): The sampling rate of the data.
            - "channels" (list): The list of channel names.
    """
    meta_info = {}
    edf = read_raw_edf(psg_file_path, preload=False, stim_channel=None, verbose=False)
    meta_info["sample_rate"] = int(edf.info["sfreq"])
    meta_info["channels"] = edf.info["ch_names"]
    data = edf.get_data().T  # Transpose data to have shape (n_samples, n_channels)
    return data, meta_info


def extract_s_id(file_path: str) -> str:
    """
    Extracts the subject ID from the given file path.

    The function assumes the file path follows a specific structure:
    - /home/xxx/data/isruc/isruc-sg1/1/1.rec
    - /home/xxx/data/isruc/isruc-sg1/1/1_1.txt
    - /home/xxx/data/isruc/isruc-sg2/1/2/2.rec
    - /home/xxx/data/isruc/isruc-sg3/1/1.rec

    :param file_path: The file path to extract the subject ID from.
    :return: A string representing the subject ID in the format "sgX-subject-visit".
    """
    path_elements = file_path.split("/")

    # Find the index of the element containing "isruc-sg"
    sgrp_idx = [i for i, x in enumerate(path_elements) if "isruc-sg" in x][0]

    # Extract the subject group (sg1, sg2, sg3, etc.)
    sgrp = path_elements[sgrp_idx].split("-")[-1]

    # The subject ID is the element following the subject group
    s_id_idx = sgrp_idx + 1
    s_id = path_elements[s_id_idx]

    # For sg2, the visit number is the next element; otherwise, it's always 1
    visit = path_elements[s_id_idx + 1] if sgrp == "sg2" else 1

    return f"{sgrp}-{s_id}-{visit}"


def read_files(signals_dir: str) -> dict:
    """
    Reads the signal and annotation files from the given directory and organizes them by subject visits.

    :param signals_dir: Directory containing the PSG and annotation files.
    :return: A dictionary where keys are subject visit identifiers and values are tuples containing the signal file path and the annotation file path.
    """
    sig_files = glob(expanduser(join(signals_dir, "**", "*.rec")), recursive=True)
    # Find all _1.txt files (first annotator's annotation files)
    sta_1_files = glob(expanduser(join(signals_dir, "**", "*_1.txt")), recursive=True)
    # Find all _2.txt files (second annotator's annotation files)
    sta_2_files = glob(expanduser(join(signals_dir, "**", "*_2.txt")), recursive=True)

    # Extract unique subject visit identifiers from the file paths
    subject_visits = list(
        set(
            [extract_s_id(f) for f in sig_files]
            + [extract_s_id(f) for f in sta_1_files]
            + [extract_s_id(f) for f in sta_2_files]
        )
    )
    assert len(sig_files) == len(subject_visits)

    # Create a dictionary mapping each subject visit to its corresponding signal and annotation files
    # Use the first annotator's annotation file if available; otherwise, use the second annotator's annotation file
    files = {
        s: (
            [f for f in sig_files if extract_s_id(f) == s][0],
            next(
                (f for f in sta_1_files if extract_s_id(f) == s),
                next(f for f in sta_2_files if extract_s_id(f) == s),
            ),
        )
        for s in subject_visits
    }

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signals_dir",
        "-s",
        type=str,
        required=True,
        help="File path to the PSG files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Directory where to save numpy files outputs.",
    )
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    files = read_files(args.signals_dir)
    print(files)

    for s_id in files.keys():
        print(f"Processing subject {s_id}...")

        # load data for subject visit
        data, meta_info = extract_from_rec(extract_from_edf)(files[s_id][0])
        all_channels = meta_info["channels"]
        raw_df = pd.DataFrame(data, columns=all_channels)

        sig = np.zeros((raw_df.shape[0], 0))
        for ch_variation in eeg_channels + emg_channels + eog_channels:
            # check if one of the channels in the variation is in the PSG file
            ch_to_load = [ch for ch in ch_variation if ch in all_channels]
            # if none of the channels in the variation is in the PSG file, try to create the channel by linear combination
            if len(ch_to_load) == 0:
                print(f"Subject {s_id} is missing channel: {ch_variation}")
                for ec in ch_variation:
                    c_left = ec.split("-")[0]
                    c_right = ec.split("-")[1]
                    if c_left in all_channels and c_right in all_channels:
                        print(
                            f"Creating channel {ec} by linear combination of {c_left} and {c_right}"
                        )
                        sig = np.concatenate(
                            (
                                sig,
                                (raw_df[c_left] - raw_df[c_right]).to_numpy()[:, None],
                            ),
                            axis=1,
                        )
            else:
                sig = np.concatenate((sig, raw_df[ch_to_load].to_numpy()), axis=1)

        if not sig.shape[1] == len(eeg_channels) + len(emg_channels) + len(
            eog_channels
        ):
            print(
                f"Subject {s_id} has {sig.shape[1]} channels, "
                f"expected {len(eeg_channels) + len(emg_channels) + len(eog_channels)}\n"
                f"this subject will be skipped"
            )
            continue
        sr = meta_info["sample_rate"]

        sig_new = []
        # downsample and filter signals
        for i, c in enumerate(eeg_channels + emg_channels + eog_channels):
            # low- and highpass filters from AASM
            sig_new.append(
                downsample(
                    sig[:, i], sr_old=sr, sr_new=SAMPLING_RATE, fmin=0.3, fmax=35
                )
            )
        sig_new = np.array(sig_new)
        # Verify that we can split into 30-s epochs
        if sig_new.shape[1] % (EPOCH_SEC_SIZE * SAMPLING_RATE) != 0:
            raise Exception(
                f"s_id {s_id} cannot be split into 30-s epochs, "
                f"num_epochs={sig_new.shape[1] / (EPOCH_SEC_SIZE * SAMPLING_RATE)}"
            )
        n_epochs = sig_new.shape[1] / (EPOCH_SEC_SIZE * SAMPLING_RATE)
        # split signals into epochs
        x = np.asarray(np.split(sig_new, n_epochs, axis=1)).astype(np.float32)
        x = x.transpose((1, 0, 2))  # we want to have (n_channels, n_epochs, n_samples)

        hypnogram = extract_from_txt(files[s_id][1])
        assert x.shape[1] == len(
            hypnogram
        ), f"Length of Hypnogram {len(hypnogram)} does not match the number of epochs {x.shape[1]}"

        save_dict = {
            ch[0]: x[i, :, :]
            for i, ch in enumerate(eeg_channels + emg_channels + eog_channels)
        }
        save_dict["y"] = hypnogram
        np.savez(join(args.output_dir, f"{s_id}.npz"), **save_dict)


if __name__ == "__main__":
    main()
