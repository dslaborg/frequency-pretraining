import os
from logging import getLogger
from os.path import join

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch import nn

from base.config import Config

logger = getLogger(__name__)


def _weights_init(m: nn.Module):
    """
    Initialize weights of the model using He-initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for layer_p in m._all_weights:
            for p in layer_p:
                if "weight" in p:
                    nn.init.kaiming_normal_(
                        m.__getattr__(p), mode="fan_out", nonlinearity="relu"
                    )
                if "bias" in p and m.__getattr__(p) is not None:
                    m.__getattr__(p).data.fill_(0)


class BaseModel(nn.Module):
    # identifier for the model, is used to create the snapshot file name
    identifier = "base_model"

    def __init__(self, seed: int):
        """
        Base class for all models with some common functionality

        :param seed: seed for random number generators used for model initialization and other random operations
        """
        super(BaseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def save(self, epoch: int = None):
        """
        Save the model to a file. The file name is created using the config name, the model identifier and the current
        timestamp. If the epoch is given, it is added to the file name. Also see create_save_file_name.
        """
        model_state = {"state_dict": self.state_dict()}
        snapshot_file = self.create_save_file_name(epoch)

        # ensure that the directory exists
        snapshot_dir = os.path.dirname(snapshot_file)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        torch.save(model_state, snapshot_file)
        logger.info(f"snapshot saved to {snapshot_file}")

    def create_save_file_name(self, epoch: int = None) -> str:
        """
        Create the file name for the snapshot file. The file name is created using the config name, the model identifier
        and the current timestamp. If the epoch is given, it is added to the file name.
        """
        _hydra_cfg = HydraConfig.get()
        _cfg = Config.get()

        snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
        config_name = _hydra_cfg.job.config_name.split("/")[-1]
        timestamp = _hydra_cfg.run.dir.split("/")[-1]

        save_file_name = f"{config_name}"
        if epoch is not None:
            save_file_name += f"-e{epoch}"
        if not OmegaConf.is_missing(_hydra_cfg.job, "num"):
            save_file_name += f"-m{_hydra_cfg.job.num}"
        save_file_name += f"-{self.identifier}-{timestamp}-final.pth"

        return join(snapshot_dir, save_file_name)

    def load(self, path: str):
        """
        Load the model from a file given by the path. If the path is None, the model is not loaded.
        """
        if path is not None:
            _cfg = Config.get()
            snapshot_dir = to_absolute_path(_cfg.general.snapshot_dir)
            model_state = torch.load(join(snapshot_dir, path))
            self.load_state_dict(model_state["state_dict"])

    def get_parameters(self, lr: float) -> list[dict]:
        """get the parameters of the model for the optimizer"""
        parameters = [{"params": self.parameters()}]
        return parameters
