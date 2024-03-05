"""
This script prepares the DODO dataset for the dataloader. It reads the PSG files and the annotation files and saves the
signals and the hypnograms in numpy files. The signals are downsampled to 100 Hz and filtered with a low- and highpass
filter. The hypnograms are consensus hypnograms from all scorers.
"""
import argparse
import glob
import os
import shutil
import sys
from glob import glob
from os.path import basename
from os.path import join, realpath, dirname

import h5py
import numpy as np

from consensus import ConsensusBuilder

sys.path.insert(0, realpath(join(dirname(__file__), "../..")))
from preprocessing.downsample import downsample

# load all EEG, EMG and EOG channels
eeg_channels = ["C3_M2", "C4_M1", "F3_F4", "F3_M2", "F3_O1", "F4_O2", "O1_M2", "O2_M1"]
emg_channels = ["EMG"]
eog_channels = ["EOG1", "EOG2"]
# artefacts are marked as -1
NOT_SCORED = -1
# epoch size in seconds
EPOCH_SEC_SIZE = 30
# sampling rate to downsample to
SAMPLING_RATE = 100


def read_files(signals_dir: str = None, annotations_dir: str = None):
    """
    Read the PSG and annotation files and return a dictionary with the PSG files as keys and the annotation files as
    values.

    :param signals_dir: file path to the PSG files
    :param annotations_dir: file path to the annotation files
    """
    sig_files = glob(join(signals_dir, "*.h5"))
    sta_files = glob(join(annotations_dir, "scorer_*/*.json"))

    subjects = np.intersect1d(
        [basename(f)[:-3] for f in sig_files], [basename(f)[:-5] for f in sta_files]
    )
    assert np.all([len([f for f in sig_files if s in f]) == 1 for s in subjects])
    assert np.all([len([f for f in sta_files if s in f]) == 5 for s in subjects])

    files = {
        [f for f in sig_files if s in f][0]: sorted([f for f in sta_files if s in f])
        for s in subjects
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
        "--annotations_dir",
        "-a",
        type=str,
        required=True,
        help="File path to the annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="cache/dodo",
        help="Directory where to save numpy files outputs.",
    )
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    files = read_files(args.signals_dir, args.annotations_dir)
    consensus_builder = ConsensusBuilder(args.annotations_dir)
    consensus_all_scorers = consensus_builder.result_hypnograms_consensus

    for sig_file in files:
        with h5py.File(sig_file, "r") as h5f:
            s_id = basename(sig_file)[:-3]

            signals = h5f["signals"]
            # concatenate all EEG, EMG and EOG channels into one matrix
            eeg = signals["eeg"]
            emg = signals["emg"]
            eog = signals["eog"]
            sig = np.concatenate([eeg[c][:][None, :] for c in eeg_channels], axis=0)
            sig = np.concatenate(
                (
                    sig,
                    np.concatenate([emg[c][:][None, :] for c in emg_channels], axis=0),
                ),
                axis=0,
            )
            sig = np.concatenate(
                (
                    sig,
                    np.concatenate([eog[c][:][None, :] for c in eog_channels], axis=0),
                ),
                axis=0,
            )
            sig_new = []
            # downsample and filter signals
            for i, c in enumerate(eeg_channels + emg_channels + eog_channels):
                # low- and highpass filters from AASM
                sig_new.append(
                    downsample(
                        sig[i],
                        sr_old=eeg.attrs["fs"],
                        sr_new=SAMPLING_RATE,
                        fmin=0.3,
                        fmax=35,
                    )
                )
            sig_new = np.array(sig_new)
            # Verify that we can split into 30-s epochs
            if sig_new.shape[1] % (EPOCH_SEC_SIZE * SAMPLING_RATE) != 0:
                raise Exception(
                    f"{sig_file} cannot be split into 30-s epochs, "
                    f"num_epochs={sig_new.shape[1] / (EPOCH_SEC_SIZE * SAMPLING_RATE)}"
                )
            n_epochs = sig_new.shape[1] / (EPOCH_SEC_SIZE * SAMPLING_RATE)
            # split signals into epochs
            x = np.asarray(np.split(sig_new, n_epochs, axis=1)).astype(np.float32)
            x = x.transpose(
                (1, 0, 2)
            )  # we want to have (n_channels, n_epochs, n_samples)

            gs_hypnogram = consensus_all_scorers[s_id][0]

            if (
                gs_hypnogram.shape[0]
                != sig_new.shape[1] // SAMPLING_RATE // EPOCH_SEC_SIZE
            ):
                print(
                    f"{sig_file} signals mismatch scores {sig_new.shape[1] // SAMPLING_RATE // EPOCH_SEC_SIZE} "
                    f"!= {gs_hypnogram.shape[0]}"
                )
                continue

            # save signals and hypnograms
            # file format: {channel: (n_epochs, n_samples), y: (n_epochs,)}
            save_dict = {
                ch: x[i, :, :]
                for i, ch in enumerate(eeg_channels + emg_channels + eog_channels)
            }
            save_dict["y"] = gs_hypnogram
            np.savez(join(args.output_dir, f"{s_id}.npz"), **save_dict)


if __name__ == "__main__":
    main()
