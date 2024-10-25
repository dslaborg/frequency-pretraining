"""
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
"""

import argparse
import glob
import math
import ntpath
import os
import shutil
import sys
from datetime import datetime
from os.path import join, realpath, dirname

import numpy as np
from mne.io import read_raw_edf

import edf_reader

sys.path.insert(0, realpath(join(dirname(__file__), "../..")))
from preprocessing.downsample import downsample

# Label values
UNKNOWN = 5

stage_dict = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "UNKNOWN": UNKNOWN}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}

channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

EPOCH_SEC_SIZE = 30
SAMPLING_RATE = 100
REMOVE_WAKE_30MIN = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        required=True,
        help="File path to the PSG and annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="cache/sleep-edfx",
        help="Directory where to save numpy files outputs.",
    )
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info["sfreq"]
        raw_ch_df = raw.to_data_frame()[channels]
        # raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # Get raw header
        f = open(psg_fnames[i], "r", errors="ignore")
        reader_raw = edf_reader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw["date_time"], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fnames[i], "r", errors="ignore")
        reader_ann = edf_reader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann["date_time"], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []  # indices of the data that will be removed
        labels = []  # indices of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str[2:-1]]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=int
                )
                label_idx.append(idx)

                print(
                    "Include onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    )
                )
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=int
                )
                remove_idx.append(idx)

                print(
                    "Remove onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    )
                )
        labels = np.hstack(labels)

        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print(
                "before remove extra labels: {}, {}".format(
                    select_idx.shape, labels.shape
                )
            )
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                # n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                # n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                n_label_trims = int(
                    math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate))
                )
                if n_label_trims != 0:
                    # select_idx = select_idx[:-n_trims]
                    labels = labels[:-n_label_trims]
            print(
                "after remove extra labels: {}, {}".format(
                    select_idx.shape, labels.shape
                )
            )

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # downsample and filter data
        old_raw_ch = raw_ch.copy()
        raw_ch = np.empty(
            (
                int(old_raw_ch.shape[0] / sampling_rate * SAMPLING_RATE),
                old_raw_ch.shape[1],
            )
        )
        for j, ch in enumerate(channels):
            raw_ch[:, j] = downsample(
                old_raw_ch[:, j],
                sr_old=sampling_rate,
                sr_new=SAMPLING_RATE,
                fmin=0.3,
                fmax=35,
            )

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * SAMPLING_RATE) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * SAMPLING_RATE)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        if REMOVE_WAKE_30MIN:
            # Select only sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y != stage_dict["W"])[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(y):
                end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx + 1)
            print("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {ch: x[:, :, i] for i, ch in enumerate(channels)}
        save_dict["y"] = y
        save_dict["fs"] = SAMPLING_RATE
        save_dict["header_raw"] = h_raw
        save_dict["header_annotation"] = h_ann

        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        print("\n=======================================\n")


if __name__ == "__main__":
    main()
