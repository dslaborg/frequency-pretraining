import torch.nn as nn

from base.config import Config
from base.model.base_featureextractor import BaseFeatureExtractor
from base.model.base_model import _weights_init
from base.model.utilities import pad_for_same


class FeTinySleepNet(BaseFeatureExtractor):
    identifier = "fe_tinysleepnet"

    def __init__(self, filters: int, dropout: list[float], path=None, seed: int = None):
        """
        Feature extractor for TinySleepNet,
        based on: https://github.com/akaraspt/tinysleepnet/blob/main/model.py and
        https://github.com/akaraspt/tinysleepnet/blob/main/nn.py

        :param filters: number of filters in each convolutional layer
        :param dropout: list of two dropout rates, first dropout layer is after the first maxpooling layer, second dropout
        layer is after the last convolutional layer
        :param path: path to load model from
        :param seed: random seed for initialization
        """
        super(FeTinySleepNet, self).__init__(seed)
        self.filters = filters
        self.dropout = dropout

        _cfg = Config.get()
        # get the number of channels and the sampling rate from the dataset used for training
        num_channels = len(_cfg.data.downstream.train_dataloader.dataset.channels)
        sampling_rate = _cfg.data.sampling_rate
        epoch_length = sampling_rate * _cfg.data.epoch_duration

        self.feature_extractor = nn.Sequential(
            # pad all convolutions so that the output has the same length as the input
            nn.ConstantPad1d(
                pad_for_same(epoch_length, sampling_rate // 2, sampling_rate // 4), 0
            ),
            nn.Conv1d(
                num_channels, self.filters, sampling_rate // 2, sampling_rate // 4
            ),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d(
                pad_for_same(epoch_length // (sampling_rate // 4), 8, 8), 0
            ),
            nn.MaxPool1d(8, 8),
            nn.Dropout(self.dropout[0]),

            nn.ConstantPad1d(
                pad_for_same(epoch_length // (sampling_rate // 4) // 8, 8), 0
            ),
            nn.Conv1d(self.filters, self.filters, 8),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d(
                pad_for_same(epoch_length // (sampling_rate // 4) // 8, 8), 0
            ),
            nn.Conv1d(self.filters, self.filters, 8),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d(
                pad_for_same(epoch_length // (sampling_rate // 4) // 8, 8), 0
            ),
            nn.Conv1d(self.filters, self.filters, 8),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(inplace=True),
            
            nn.ConstantPad1d(
                pad_for_same(epoch_length // (sampling_rate // 4) // 8, 4, 4), 0
            ),
            nn.MaxPool1d(4, 4),
            nn.Dropout(self.dropout[1]),
        )

        self.apply(_weights_init)

        self.load(path)  # always load model at the end of the initialization
