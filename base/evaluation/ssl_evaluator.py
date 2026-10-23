from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.results.result_tracker import ResultTracker

logger = getLogger(__name__)


class SSLEvaluator(BaseEvaluator):
    def evaluate(self, model: nn.Module):
        _cfg = Config.get()
        result_tracker = ResultTracker.get(self.dataset)

        model.eval()
        loss = torch.zeros(1).to(_cfg.general.device)
        loss_components = []

        with torch.no_grad():
            for data in self.dataloader:
                features_1, features_2, _ = data
                features_1 = features_1.to(_cfg.general.device)
                features_2 = features_2.to(_cfg.general.device)

                loss_i, loss_components_i = model((features_1, features_2))

                loss += loss_i * features_1.size(0)
                loss_components.append(loss_components_i)

        result_tracker.update_ssl_loss(loss.item() / len(self.dataloader.dataset))
        result_tracker.update_ssl_loss_components(loss_components)

    def log(self):
        result_tracker = ResultTracker.get(self.dataset)
        logger.info(
            f'dataset {self.dataset}, '
            f'current loss components: {[f"{a:.3f}" for a in result_tracker.metrics["ssl"]["loss_components"][-1]]}'
        )
