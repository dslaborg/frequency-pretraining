from logging import getLogger

import torch
import torch.nn as nn
from torch.nn import functional as F

from base.model.base_model import _weights_init, BaseModel
from base.model.pretraining.base_pret_fe import BasePretrainingFeatureExtractor

logger = getLogger(__name__)


class SimClrProjector(BaseModel):
    identifier = "simclr_projector"

    def __init__(self, feature_size, projector_dims, path=None, seed: int = None):
        self.feature_size = eval(str(feature_size))  # allows configs like '128 * 4'
        self.projector_dims = projector_dims
        super(SimClrProjector, self).__init__(seed)

        layers = [nn.Linear(self.feature_size, self.projector_dims[0])]
        for i in range(len(self.projector_dims) - 1):
            layers.append(nn.BatchNorm1d(self.projector_dims[i])),
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.projector_dims[i], self.projector_dims[i + 1]))
        self.projector = nn.Sequential(*layers)

        self.apply(_weights_init)

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        out = self.projector(x)
        return out


class SimClrLoss(torch.nn.Module):
    """credits to https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py"""
    def __init__(self, temperature):
        super(SimClrLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.

        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim],
                                     where features[:batch_size] are the representations of
                                     the first set of augmented images, and features[batch_size:]
                                     are the representations of the second set.

        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        # Coscine similarity is computed as dot product of normalized vectors
        features = F.normalize(features, dim=1)

        # Compute similarity matrix, shape [2 * batch_size, 2 * batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2

        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float("inf"))

        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss, [loss.item()]


class SimClrModel(BasePretrainingFeatureExtractor):
    identifier = "simclr"

    def __init__(
        self,
        encoder,
        encoding_size,
        projector_dims,
        is_pretraining,
        temperature,
        path=None,
        seed: int = None,
    ):
        super(SimClrModel, self).__init__(is_pretraining, seed)
        self.encoder = encoder

        self.projector = SimClrProjector(encoding_size, projector_dims)
        self.loss = SimClrLoss(temperature)

        self.apply(_weights_init)

        self.load(path)  # always load model at the end of the initialization

    def forward_train(self, x):
        # concatenate along batch dim
        x = torch.cat(x, dim=0)

        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.projector(x)

        return self.loss(x)

    def forward_eval(self, x):
        x = self.encoder(x)
        return x
