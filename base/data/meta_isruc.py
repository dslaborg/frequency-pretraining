import os
from glob import glob
from os.path import basename

import numpy as np
from hydra.utils import to_absolute_path

from base.data.meta_base import MetaBase


class MetaIsruc(MetaBase):
    def __init__(self, data_path: str):
        """
        Meta object to load the data from the ISRUC datasets. Data files need to be preprocessed and saved as .npz files
        (see preprocessing/isruc/prepare_isruc.py).

        :param data_path: path to the data folder containing the .npz files
        """
        self.data_path = data_path

        if not os.path.exists(to_absolute_path(self.data_path)):
            raise ValueError(
                f"ERROR: configured path to data folder does not exist: {to_absolute_path(self.data_path)}"
            )

    def load_data_files(self, subject_ids: list[str]) -> list[str]:
        """
        Load the data filenames for the given subject ids.

        :param subject_ids: list of subject ids to filter for

        :return: list of filenames
        """
        files = sorted(glob(os.path.join(to_absolute_path(self.data_path), "*.npz")))

        # naming scheme of data files is sg<N>-<xx>-<vv> with xx being the number of the subject (its ID)
        # and N being the number of the subgroup and vv being the visit number
        files = [
            f for subj_id in subject_ids for f in files if subj_id == basename(f)[:-4]
        ]

        return files

    def load_data(
        self, subject_ids: list[str], channels: list[str]
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
        """
        Load the data for the given subject ids and channels.

        :param subject_ids: list of subject ids to filter for
        :param channels: list of channels to load

        :return: x_data, y_data as dictionaries with the subject id as key
        """
        files = self.load_data_files(subject_ids)
        x_data = {
            basename(np_file)[:-4]: {ch: np.load(np_file)[ch] for ch in channels}
            for np_file in files
        }
        y_data = {basename(np_file)[:-4]: np.load(np_file)["y"] for np_file in files}

        # convert y_data to numpy arrays
        y_data = {k: np.array(v).astype(int) for k, v in y_data.items()}
        return x_data, y_data
