from logging import getLogger

import numpy as np
import torch
from torch import nn

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.results.result_tracker import ResultTracker

logger = getLogger(__name__)


class ClasEvaluator(BaseEvaluator):
    """Evaluator for the sleep stage classification task."""

    def evaluate(self, model: nn.Module):
        """
        Evaluate the model on the given dataset

        :param model: model to evaluate
        """
        _cfg = Config.get()
        result_tracker: ResultTracker = ResultTracker.get(self.dataset)

        model.eval()
        # aggregate predictions, actual labels, and loss over the dataset
        predictions: np.ndarray = np.empty([0, 5])
        actual_labels: np.ndarray = np.empty(0, dtype="int")
        loss: torch.Tensor = torch.zeros(1).to(_cfg.general.device)

        with torch.no_grad():
            for data in self.dataloader:
                features, labels = data
                features = features.to(_cfg.general.device)
                labels = labels[:, self.dataloader.dataset.left_epochs, :]
                labels = labels.long().flatten().to(_cfg.general.device)

                outputs = model(features)
                outputs = outputs[:, self.dataloader.dataset.left_epochs, :]

                # if the batch only contains artefacts, skip it
                if not torch.all(labels == -1):
                    criterion = nn.CrossEntropyLoss()
                    loss += criterion(outputs[labels != -1], labels[labels != -1])

                probabilities = torch.softmax(outputs, dim=1)

                predictions = np.r_["0", predictions, probabilities.tolist()]
                actual_labels = np.r_[actual_labels, labels.tolist()]

        predicted_labels = np.argmax(predictions, axis=1)

        # remove artefacts from the predictions and actual labels
        artefacts_mask = actual_labels != -1
        actual_labels = actual_labels[artefacts_mask].astype(int)
        predicted_labels = predicted_labels[artefacts_mask].astype(int)

        result_tracker.update_downstream_loss(loss.item() / len(self.dataloader))
        result_tracker.update_downstream_metrics_and_matrices(
            actual_labels, predicted_labels
        )

    def log(self):
        """
        Log the evaluation results of the latest evaluation, should be called after evaluate.
        """
        result_tracker = ResultTracker.get(self.dataset)
        logger.info(
            f"dataset: {self.dataset}, avg f1-score: {result_tracker.get_last_macro_f1_score():.4f}"
        )
