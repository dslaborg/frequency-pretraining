"""
Perform fine-tuning on a downstream task using the configured model and dataset. The model can be pretrained or
initialized randomly (depending on the model path set in the configuration).
"""
import os
import shutil
import sys
from logging import getLogger
from os.path import realpath, dirname, join

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, realpath(join(dirname(__file__), '..')))

from base.results.result_tracker import ResultTracker
from base.config import Config

logger = getLogger(__name__)


def initialize(cfg):
    """
    Initializes the configuration and the ResultTrackers. Also handles GPU selection if multiple runs are executed on
    multiple GPUs on the same machine.
    """
    if cfg.general.device == 'cuda' and not OmegaConf.is_missing(HydraConfig.get().job, 'num'):
        gpu_id = HydraConfig.get().job.num % len(cfg.general.gpus)
        gpu_id = cfg.general.gpus[gpu_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        logger.info(f'Using GPU {gpu_id}')

    Config.initialize(cfg)
    ResultTracker.initialize()


@hydra.main(config_path='../config', version_base='1.2')
def main(cfg: DictConfig):
    initialize(cfg)
    hydra_cfg = HydraConfig.get()
    logger.info(f'overrides:\n{OmegaConf.to_yaml(hydra_cfg.overrides)}')

    # only relevant for multi runs: copy the .hydra directory with the configs of the first run to the parent directory
    # .hydra folder in subdirectories can then be ignored by git to avoid cluttering the repository
    if not OmegaConf.is_missing(hydra_cfg.job, 'num') and hydra_cfg.job.num == 0:
        shutil.copytree(join(os.getcwd(), hydra_cfg.output_subdir), join(os.getcwd(), '..', hydra_cfg.output_subdir),
                        ignore=shutil.ignore_patterns('overrides.yaml'))

    # initialize the trainer object, all other objects are initialized recursively within the trainer according to the
    # configuration
    trainer = instantiate(cfg.training.downstream.trainer)
    best_macro_f1_score = trainer.train()

    # use macro f1 score as optimization criterion for nevergrad sweeps
    # cast to float, because OmegaConf doesn't like numpy datatypes
    return float(best_macro_f1_score)


if __name__ == '__main__':
    main()
