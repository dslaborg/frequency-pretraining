from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.results.result_tracker import ResultTracker

logger = getLogger(__name__)


class FreqEvaluator(BaseEvaluator):
    """Evaluator for the frequency pretraining task."""

    def evaluate(self, model: nn.Module):
        """
        Evaluate the model on the frequency pretraining task.

        :param model: The model to evaluate.
        """
        _cfg = Config.get()
        n_freqs = _cfg.data.pretraining.n_freqs
        result_tracker = ResultTracker.get(self.dataset)

        model.eval()
        # aggregate predictions, actual labels, and loss over the entire dataset
        predicted_labels = np.empty((0, n_freqs), dtype="int")
        actual_labels = np.empty((0, n_freqs), dtype="int")
        loss = torch.zeros(1).to(_cfg.general.device)

        with torch.no_grad():
            for data in self.dataloader:
                features, labels = data
                features = features.to(_cfg.general.device)
                labels = labels.float().to(_cfg.general.device)

                outputs = model(features)

                criterion = nn.BCEWithLogitsLoss()
                loss += criterion(outputs, labels) * features.shape[0]

                # since the frequency pretraining task is a multi-label classification task,
                # we use a sigmoid activation function to get the predicted labels
                # and then threshold the output at 0.5
                predicted_labels_i = (torch.sigmoid(outputs).cpu() > 0.5).long()

                predicted_labels = np.r_[predicted_labels, predicted_labels_i.numpy()]
                actual_labels = np.r_[actual_labels, labels.cpu().numpy()]

        result_tracker.update_freq_loss(loss.item() / len(self.dataloader.dataset))
        result_tracker.update_freq_metrics(actual_labels, predicted_labels)

    def log(self):
        """
        Log the frequency pretraining task evaluation results from the last evaluation, should be called after evaluate.
        """
        result_tracker = ResultTracker.get(self.dataset)
        metrics = result_tracker.get_last_freq_metrics()
        logger.info(
            f"dataset: {self.dataset}, "
            f'acc: {metrics["acc_freq"]:.3f}, '
            f'hamming: {metrics["hamming_freq"]:.3f}'
        )
