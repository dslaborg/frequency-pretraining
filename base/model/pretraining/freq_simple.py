import torch
import torch.nn as nn

from base.model.base_model import _weights_init
from base.model.pretraining.base_pret_fe import BasePretrainingFeatureExtractor


class SimpleFreqModel(BasePretrainingFeatureExtractor):
    identifier = "simple_multi_class"

    def __init__(
        self,
        encoder: nn.Module,
        encoding_size: int,
        n_outputs: int,
        is_pretraining: bool,
        seed: int = None,
        path: str = None,
    ):
        """
        Simple model architecture for the frequency pretraining task. The model consists of a modular encoder and a
        classifier. The classifier is a simple feed forward neural network with two hidden layers and ReLU activation
        functions.

        :param encoder: The encoder module, also called feature extractor.
        :param encoding_size: The size of the encoding produced by the encoder.
        :param n_outputs: The number of output classes.
        :param is_pretraining: Whether the model is used for pretraining or not.
        :param seed: The random seed used for initialization.
        :param path: The path to the model file to load. If None, the model is initialized randomly.
        """
        super(SimpleFreqModel, self).__init__(is_pretraining, seed)
        self.encoder = encoder
        self.encoding_size = encoding_size
        self.n_outputs = n_outputs

        self.classifier = nn.Sequential(
            nn.Linear(self.encoding_size, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, self.n_outputs),
        )

        self.apply(_weights_init)

        self.load(path)  # always load model at the end of the initialization

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for evaluation. The classifier is not used in this case, since the model is used for feature
        extraction only.
        """
        x = self.encoder(x)
        return x
