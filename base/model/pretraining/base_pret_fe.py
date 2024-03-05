import glob
import os
from logging import getLogger
from os.path import join

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path

from base.config import Config
from base.model.base_featureextractor import BaseFeatureExtractor

logger = getLogger(__name__)


class BasePretrainingFeatureExtractor(BaseFeatureExtractor):
    identifier = "base_pret_fe"

    def __init__(self, is_pretraining: bool, seed: int):
        """
        Base class for feature extractors used in pretraining.

        :param is_pretraining: Whether the model is used for pretraining or fine-tuning.
        :param seed: Random seed used for initialization.
        """
        super(BasePretrainingFeatureExtractor, self).__init__(seed)
        self.is_pretraining = is_pretraining

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model. If the model is used for pretraining, the forward_train method is called,
        otherwise the forward_eval method is called.

        :param x: Input tensor.

        :return: Output tensor.
        """
        if self.is_pretraining:
            return self.forward_train(x)
        else:
            return self.forward_eval(x)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model during pretraining.
        """
        raise NotImplementedError

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model during fine-tuning.
        """
        raise NotImplementedError

    def load(self, path: str):
        """
        Load model from file.

        :param path: Path to the model file. If path is None, the model is not loaded. If path is 'same_run', the model
                        is loaded from the filename that is used to save the model in the current run. If path starts
                        with 'latest_run', the model is loaded from the latest model file in the snapshot directory. If
                        path is a valid path, the model is loaded from the specified file.
        """
        if path is None:
            path = None
        elif path == 'same_run':
            path = self.create_save_file_name()
        elif path.startswith('latest_run'):
            _hydra_cfg = HydraConfig.get()
            _cfg = Config.get()
            snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
            config_name = _hydra_cfg.job.config_name.split('/')[-1]

            model_search_str = f'{config_name}-*{self.identifier}-*.pth'
            if path.startswith('latest_run_epoch_'):
                epoch = int(path.split('_')[-1])
                model_search_str = f'{config_name}-e{epoch}-*{self.identifier}-*.pth'
            search_path = join(snapshot_dir, model_search_str)
            possible_paths = glob.glob(search_path)
            # sort files by modification time and take the last one
            path = sorted(possible_paths, key=lambda x: os.stat(x).st_mtime)[-1]

        if path is not None:
            logger.info(f'Loading model from {path}')
        super(BasePretrainingFeatureExtractor, self).load(path)
