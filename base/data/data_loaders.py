from logging import getLogger

import numpy as np
from torch.utils.data import Dataset

from base.config import Config
from base.data.data_reducer import BaseDataReducer, get_class_distribution
from base.data.meta_dod import MetaBase
from base.data.normalization import normalize_epoch

logger = getLogger(__name__)


class DatasetForSequences(Dataset):
    def __init__(
        self,
        subject_ids: list[str],
        meta_obj: MetaBase,
        channels: list[str],
        left_epochs: int,
        right_epochs: int,
        data_reducer: BaseDataReducer,
    ):
        """
        Dataset for sequences of EEG epochs. A sample consists of a sequence of EEG epochs and the corresponding labels
        for each epoch.
        Each Sequence consists of left_epochs + 1 + right_epochs epochs.

        :param subject_ids: list of subject ids to include in the dataset; used to control which subjects are used for
        training, validation and testing
        :param meta_obj: object that handles the loading of the data, should be replaced for different datasets
        :param channels: list of channels to load
        :param left_epochs: number of epochs to include on the left side of the middle epoch in a sequence
        :param right_epochs: number of epochs to include on the right side of the middle epoch in a sequence
        :param data_reducer: object that handles the reduction of the data for experiments with a limited amount of
        samples
        """
        super(DatasetForSequences, self).__init__()
        self.meta_obj = meta_obj
        self.channels = channels
        self.left_epochs = left_epochs
        self.right_epochs = right_epochs
        self.data_reducer = data_reducer

        # load and normalize data
        _x_data, self.y_data = self.meta_obj.load_data(subject_ids, self.channels)
        # format _x_data: (subj_id: channel: [epoch, datapoints])
        self.x_data = {}
        for s_id in _x_data:
            self.x_data[s_id] = np.concatenate(
                [_x_data[s_id][ch][:, np.newaxis, :] for ch in self.channels],
                axis=1,
                dtype="float32",
            )
            self.x_data[s_id] = normalize_epoch(self.x_data[s_id])
        # format self.x_data: (subj_id: [epoch, channel, datapoints])

        # create indices for the dataset, each index is a tuple (subject_id, start_index, end_index) and represents a
        # sequence of epochs (sample)
        # we can use all epochs as "middle" epochs, the epochs at the start and the end of a recording must be buffered
        self.indices = [
            (
                s_id,
                idx - self.left_epochs,
                idx + self.right_epochs + 1,
            )  # idx_end is exclusive
            for s_id, s_data in self.x_data.items()
            for idx in range(len(s_data))
        ]
        self.indices = sort_indices(self.indices)
        logger.info(
            f"class distribution before data reduction:\n"
            f"{get_class_distribution(self.indices, self.y_data, self.left_epochs)}"
        )
        self.indices = self.data_reducer.reduce_data(self.indices, self.y_data)
        self.indices = sort_indices(self.indices)
        logger.info(
            f"class distribution after data reduction:\n"
            f"{get_class_distribution(self.indices, self.y_data, self.left_epochs)}"
        )
        self.len = len(self.indices)

        # number of datapoints in an epoch
        self.len_epoch = self.x_data[self.indices[0][0]].shape[-1]

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset. index[0] is the subject id and determines which subject's data is used.
        The sample consists of a sequence of EEG epochs starting at index[1] and ending at index[2] (exclusive) and
        the corresponding labels for each epoch.

        :return: tuple of the form (x, y) where x is a tensor of shape (left_epochs + 1 + right_epochs, n_channels,
        datapoints)
        """
        _cfg = Config.get()
        s_id, idx_start, idx_end = self.indices[index]

        # buffer left and right epochs with zeros if idx_start < 0 or idx_end > len(x_data[...])
        left_epochs = np.empty((0, len(self.channels), self.len_epoch))
        right_epochs = np.empty((0, len(self.channels), self.len_epoch))
        left_labels = []
        right_labels = []
        if idx_start < 0:
            left_epochs = np.zeros(
                (-idx_start, len(self.channels), self.len_epoch), dtype="float32"
            )
            left_labels = [-1] * -idx_start
            idx_start = 0
        if idx_end > self.x_data[s_id].shape[0]:
            right_epochs = np.zeros(
                (
                    idx_end - self.x_data[s_id].shape[0],
                    len(self.channels),
                    self.len_epoch,
                ),
                dtype="float32",
            )
            right_labels = [-1] * (idx_end - self.x_data[s_id].shape[0])
            idx_end = self.x_data[s_id].shape[0]

        # build sequence tensors of shape (left_epochs + 1 + right_epochs, channel, datapoints)
        x = self.x_data[s_id][idx_start:idx_end]
        x = np.concatenate([left_epochs, x, right_epochs], axis=0, dtype="float32")

        y = self.y_data[s_id][idx_start:idx_end]
        y = np.concatenate([left_labels, y, right_labels])

        return x, y

    def __len__(self):
        return self.len


def sort_indices(indices: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Sort indices by subject id and index start."""
    return sorted(indices, key=lambda x: (x[0], x[1]))
