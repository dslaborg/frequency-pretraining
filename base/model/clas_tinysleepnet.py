import torch
import torch.nn as nn

from base.config import Config
from base.model.base_model import BaseModel, _weights_init


class ClasTinySleepNet(BaseModel):
    identifier = "clas_tinysleepnet"

    def __init__(self, feature_size: int | str, hidden_size: int, dropout: float, bidirectional: bool, path=None, seed: int = None):
        """
        classifier of the TinySleepNet model

        :param feature_size: number of features extracted by the feature extractor, can be an expression like '128 * 4'
        :param hidden_size: number of hidden units in the LSTM
        :param dropout: dropout rate, applied after the LSTM
        :param bidirectional: if True, the LSTM is bidirectional
        :param path: path to a model snapshot
        :param seed: random seed for initialization
        """
        super(ClasTinySleepNet, self).__init__(seed)
        self.feature_size = eval(str(feature_size))  # allows configs like '128 * 4'
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, num_layers=1, batch_first=True,
                            bidirectional=self.bidirectional)

        self.classifier = nn.Sequential(
            # nn.ReLU(inplace=True),  # apparently no activation here, seems to be part of the LSTM
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), 5)
        )

        self.apply(_weights_init)

        self.load(path)  # always load model at the end of the initialization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _cfg = Config.get()
        # input shape (batch, sequence_length, *[feature map size])
        # output shape is (batch, 5)
        batch = x.shape[0]
        sequence_length = x.shape[1]
        # flatten feature maps
        x = x.reshape(batch, sequence_length, -1)
        # initial hidden state and cell state are zero if not provided
        out, (_, _) = self.lstm(x)
        # shape (batch, 25, (1 or 2)*hidden_size)

        out = out.reshape([batch * sequence_length, -1])
        logits = self.classifier(out)
        return logits.reshape([batch, sequence_length, -1])
