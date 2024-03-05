import torch

from base.model.base_model import BaseModel


class BaseFeatureExtractor(BaseModel):
    identifier = "base_featureextractor"

    def __init__(self, seed: int):
        """
        Base class for feature extractors

        :param seed: random seed for initialization
        """
        super(BaseFeatureExtractor, self).__init__(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method to forward the input through the feature extractor. The method is able to handle
        input with 3 or 4 dimensions. If the input has 4 dimensions, the feature extractor is applied to each
        epoch in the sequence separately (axis=1). If the input has only 3 dimensions, there is only one epoch.
        """
        # if input has 4 dims, we have a sequence of epochs, shape (batch, sequence length, channels, data points)
        # if input has only 3 dims, there is only one epoch, shape (batch, channels, data points)

        if len(x.shape) == 4:
            # input shape (batch, 25, 2, 3000)
            # output shape (batch, 25, 128, 4)
            # apply feature extractor to each epoch in the sequence separately (axis=1)
            # by combining axes 0 and 1 (batch and sequence) before forwarding
            x2 = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
            out = self.feature_extractor(x2)
            # reshape back to 4 dims
            out = out.reshape(x.shape[0], x.shape[1], out.shape[1], out.shape[2])
            return out
        elif len(x.shape) == 3:
            return self.feature_extractor(x)
        else:
            raise ValueError(
                f"input to forward() of BaseFeatureExtractor has {len(x.shape)} dims "
                f"and can not be processed"
            )
