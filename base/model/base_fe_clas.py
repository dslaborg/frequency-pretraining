import glob
import os
from logging import getLogger
from os.path import join

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from torch import nn

from base.config import Config
from base.model.base_model import BaseModel

logger = getLogger(__name__)


class BaseFeClasModel(BaseModel):
    identifier = "base_fe_clas"

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        finetune_feature_extractor: bool,
        seed: int,
        path: str = None,
    ):
        """
        model combining a generic feature extractor and a classifier

        :param feature_extractor: feature extractor
        :param classifier: classifier
        :param finetune_feature_extractor: if True, the feature extractor is finetuned, otherwise it is fixed
        :param seed: random seed for initialization
        :param path: path to a model file to load
        """
        super(BaseFeClasModel, self).__init__(seed)
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.finetune_feature_extractor = finetune_feature_extractor

        # if finetune_feature_extractor is set to False remove the feature extractor from gradient calculation
        # this solves the error: "cudnn RNN backward can only be called in training mode" when training the classifier
        # with a fixed feature extractor (feature extractor is set to eval mode)
        # otherwise, removing the gradient calculation of the feature extractor is not needed, since the
        # feature extractor is not updated anyway (parameters not in optimizer)
        if not self.finetune_feature_extractor:
            self.feature_extractor.requires_grad_(False)

        self.load(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def train(self, *args, **kwargs):
        """
        Set the model to train mode. If finetune_feature_extractor is set to False, the feature extractor is set to eval
        mode to disable dropout layers, etc.
        """
        if self.finetune_feature_extractor:
            self.feature_extractor.train(*args, **kwargs)
        else:
            self.feature_extractor.eval()
        self.classifier.train(*args, **kwargs)

    def eval(self):
        self.feature_extractor.eval()
        self.classifier.eval()

    def to(self, *args, **kwargs):
        self.feature_extractor.to(*args, **kwargs)
        self.classifier.to(*args, **kwargs)

    def get_parameters(self, lr: float | list[float]) -> list[dict]:
        """
        Helper method to get the parameters for the optimizer. If finetune_feature_extractor is set to True, the
        parameters of the feature extractor and the classifier are returned. Otherwise, only the parameters of the
        classifier are returned. This method allows to set different learning rates for the feature extractor and the
        classifier.
        """
        # if lr is a single value, use the same learning rate for the feature extractor and the classifier
        if isinstance(lr, float):
            lr = [lr, lr]
        parameters = []
        if self.finetune_feature_extractor:
            parameters.append(
                {"params": self.feature_extractor.parameters(), "lr": lr[0]}
            )
        parameters.append({"params": self.classifier.parameters(), "lr": lr[1]})
        return parameters

    def load(self, path: str = None):
        """
        Load a model from a file. If path is set to None, the model is not loaded. If path is set to "latest_run", the
        latest model file is loaded. If path is set to a specific file, this file is loaded.
        """
        if path is None:
            path = None
        elif path == "latest_run":
            _hydra_cfg = HydraConfig.get()
            _cfg = Config.get()
            snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
            config_name = _hydra_cfg.job.config_name.split("/")[-1]

            model_search_str = f"{config_name}-*{self.identifier}-*.pth"
            search_path = join(snapshot_dir, model_search_str)
            possible_paths = glob.glob(search_path)
            # sort files by modification time and take the last one
            path = sorted(possible_paths, key=lambda x: os.stat(x).st_mtime)[-1]

        if path is not None:
            logger.info(f"Loading model from {path}")
        super(BaseFeClasModel, self).load(path)
