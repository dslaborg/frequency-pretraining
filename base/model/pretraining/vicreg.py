from logging import getLogger

import torch
import torch.nn as nn
from torch.nn import functional as F

from base.model.base_model import _weights_init, BaseModel
from base.model.pretraining.base_pret_fe import BasePretrainingFeatureExtractor
from base.model.utilities import off_diagonal

logger = getLogger(__name__)


class VICRegExpander(BaseModel):
    identifier = "vicreg_expander"

    def __init__(self, feature_size, expander_dims, path=None, seed: int = None):
        self.feature_size = eval(str(feature_size))  # allows configs like '128 * 4'
        self.expander_dims = expander_dims
        super(VICRegExpander, self).__init__(seed)

        layers = [nn.Linear(self.feature_size, self.expander_dims[0])]
        for i in range(len(self.expander_dims) - 1):
            layers.append(nn.BatchNorm1d(self.expander_dims[i])),
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.expander_dims[i], self.expander_dims[i + 1]))
        self.expander = nn.Sequential(*layers)

        self.apply(_weights_init)

        # always load model at the end of the initialization
        self.load(path)

    def forward(self, x):
        out = self.expander(x)
        return out


class VICRegLoss(torch.nn.Module):
    def __init__(self, lam, mu, nu, gamma, epsilon):
        super(VICRegLoss, self).__init__()
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, output1, output2):
        invariance_loss = F.mse_loss(output1, output2)

        s1 = torch.sqrt(output1.var(dim=0) + self.epsilon)
        s2 = torch.sqrt(output2.var(dim=0) + self.epsilon)

        variance_loss = F.relu(self.gamma - s1).mean() + F.relu(self.gamma - s2).mean()

        n1 = output1 - output1.mean(dim=0)
        n2 = output2 - output2.mean(dim=0)

        cov1 = (n1.T @ n1) / (output1.shape[0] - 1)
        cov2 = (n2.T @ n2) / (output2.shape[0] - 1)

        c1 = off_diagonal(cov1).pow(2).sum() / output1.shape[1]
        c2 = off_diagonal(cov2).pow(2).sum() / output2.shape[1]

        covariance_loss = c1 + c2

        return (
            self.lam * invariance_loss
            + self.mu * variance_loss
            + self.nu * covariance_loss,
            (invariance_loss.item(), variance_loss.item(), covariance_loss.item()),
        )


class VICRegModel(BasePretrainingFeatureExtractor):
    identifier = "vicreg"

    def __init__(
        self,
        encoder,
        encoding_size,
        expander_dims,
        is_pretraining,
        loss_params,
        path=None,
        seed: int = None,
    ):
        super(VICRegModel, self).__init__(is_pretraining, seed)
        self.encoder = encoder

        self.expander = VICRegExpander(encoding_size, expander_dims)
        self.loss = VICRegLoss(**loss_params)

        self.apply(_weights_init)

        self.load(path)  # always load model at the end of the initialization

    def forward_train(self, x):
        batch_size = x[0].shape[0]
        # concatenate along batch dim
        x = torch.cat(x, dim=0)

        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.expander(x)

        return self.loss(x[:batch_size], x[batch_size:])

    def forward_eval(self, x):
        x = self.encoder(x)
        return x
