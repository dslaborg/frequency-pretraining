from logging import getLogger

import numpy as np
from torch.utils.data import Dataset

from base.data.data_augmentation import BaseDataAug
from base.data.meta_base import MetaBase
from base.data.normalization import normalize_epoch

logger = getLogger(__name__)


class AugmentedDataset(Dataset):
    def __init__(
        self,
        subject_ids: dict[str, list],
        meta_obj: dict[str, MetaBase],
        channels: dict[str, list],
        data_augmentors: list[BaseDataAug],
    ):
        """
        Dataset for traditional SSL methods requiring data augmented "views" of sequences of EEG epochs. A sample consists of two view of a sequence of EEG epochs and the corresponding labels
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
        super(AugmentedDataset, self).__init__()
        self.meta_obj = meta_obj
        self.channels = channels
        self.data_augmentors = data_augmentors

        self.x_data = {}
        self.y_data = {}
        for dataset in subject_ids:
            assert dataset in self.meta_obj
            assert dataset in self.channels
            self.x_data[dataset], self.y_data[dataset] = self.meta_obj[
                dataset
            ].load_data(subject_ids[dataset], self.channels[dataset])

            # format x_data: {dataset: (subj_id: channel: [epoch, datapoints])}
            for s_id in self.x_data[dataset]:
                self.x_data[dataset][s_id] = np.concatenate(
                    [
                        self.x_data[dataset][s_id][ch][:, np.newaxis, :]
                        for ch in self.channels[dataset]
                    ],
                    axis=1,
                    dtype="float32",
                )
                self.x_data[dataset][s_id] = normalize_epoch(self.x_data[dataset][s_id], "iqr")
        # format self.x_data: {dataset: (subj_id: [epoch, channel, datapoints])}

        # create indices for the dataset, each index is a tuple (dataset, subject_id, index) and represents an epoch (sample)
        self.indices = [
            (dataset, s_id, idx)
            for dataset, d_data in self.x_data.items()
            for s_id, s_data in d_data.items()
            for idx in range(len(s_data))
        ]
        self.len = len(self.indices)

        # number of datapoints in an epoch
        self.len_epoch = self.x_data[self.indices[0][0]][self.indices[0][1]].shape[-1]

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        dataset, s_id, ep_idx = self.indices[index]

        x = self.x_data[dataset][s_id][ep_idx]

        x_aug_1 = x.copy()
        x_aug_2 = x.copy()
        for aug in self.data_augmentors:
            x_aug_1 = aug.augment(x_aug_1)
            x_aug_2 = aug.augment(x_aug_2)

        # pytorch does not like doubles
        return x_aug_1.astype(np.float32), x_aug_2.astype(np.float32)

    def __len__(self):
        return self.len
