import copy
import json
import os
from os import makedirs
from os.path import join, exists

import numpy as np
from hydra.utils import to_absolute_path
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    hamming_loss,
)

from base.config import Config
from base.results.json_utils import NumpyEncoder, remove_empty_values

KEY_MACRO = "macro"


class ResultTracker:
    """
    The ResultTracker class is used to store and write the results of the training and evaluation of the models.
    It contains a dict of ResultTracker singleton instances for each dataset that is evaluated.
    """

    __result_trackers: dict = None

    @staticmethod
    def initialize():
        """
        Initializes the ResultTracker singleton instances for each dataset that is evaluated.
        """
        _cfg = Config.get()
        # get all datasets that will be evaluated in pretraining and downstream training
        datasets = {"train"}
        if _cfg.training.pretraining.trainer:
            for evaluator in _cfg.training.pretraining.trainer.evaluators.values():
                datasets.add(evaluator.dataset)
        for evaluator in _cfg.training.downstream.trainer.evaluators.values():
            datasets.add(evaluator.dataset)

        ResultTracker.__result_trackers = {ds: ResultTracker(ds) for ds in datasets}

    @staticmethod
    def get(dataset: str) -> "ResultTracker":
        """singleton getter for ResultTracker instances"""
        if (
            ResultTracker.__result_trackers is None
            or dataset not in ResultTracker.__result_trackers
        ):
            raise ValueError(f"ResultTracker for dataset {dataset} was not initialized")
        return ResultTracker.__result_trackers[dataset]

    @staticmethod
    def write_all():
        """
        Writes the results of all ResultTracker instances to a json file.
        """
        _cfg = Config.get()
        all_metrics = {}
        if ResultTracker.__result_trackers is not None:
            for dataset, tracker in ResultTracker.__result_trackers.items():
                all_metrics[dataset] = tracker.create_dict()

        if _cfg.general.results_dir is None:
            results_dir = os.getcwd()
        else:
            results_dir = to_absolute_path(_cfg.general.results_dir)
            if not exists(results_dir):
                makedirs(results_dir)

        file = join(results_dir, f"results.json")
        with open(file, "w") as f:
            f.write(
                json.dumps(all_metrics, indent=4, sort_keys=True, cls=NumpyEncoder)
                + "\n"
            )

    def __init__(self, dataset: str):
        """
        Initializes a ResultTracker instance for a specific dataset. Each ResultTracker instance contains the metrics
        and matrices that are used to evaluate the model both during pretraining and downstream training (fine-tuning).

        :param dataset: The name of the dataset that is evaluated.
        """
        self.dataset = dataset
        _cfg = Config.get()
        self.stages = _cfg.data.stages
        self.metrics = {
            "freq": {"acc_freq": [], "hamming_freq": [], "loss": []},
            "downstream": {
                "precisions": {
                    stage: [] for stage in list(self.stages.values()) + [KEY_MACRO]
                },
                "recalls": {
                    stage: [] for stage in list(self.stages.values()) + [KEY_MACRO]
                },
                "f1_scores": {
                    stage: [] for stage in list(self.stages.values()) + [KEY_MACRO]
                },
                "loss": [],
                "cohen_kappa": [],
            },
        }
        self.markov_matrix = {"actual": [], "predicted": []}
        self.confusion_matrix = []

    def update_freq_loss(self, loss: float):
        """update the loss of the frequency prediction model"""
        self.metrics["freq"]["loss"].append(loss)

    def update_freq_metrics(self, freq_gt: np.ndarray, freq_pred: np.ndarray):
        """
        Update the accuracy and hamming metric of the frequency prediction model

        :param freq_gt: the ground truth frequency labels
        :param freq_pred: the predicted frequency labels
        """
        acc_freq = accuracy_score(freq_gt, freq_pred)
        self.metrics["freq"]["acc_freq"].append(acc_freq)
        # hamming loss see https://scikit-learn.org/stable/modules/model_evaluation.html#hamming-loss
        hamming_freq = 1 - hamming_loss(freq_gt, freq_pred)
        self.metrics["freq"]["hamming_freq"].append(hamming_freq)

    def get_last_freq_metrics(self) -> dict:
        """return the last frequency metrics as a dictionary"""
        metrics = {}
        if len(self.metrics["freq"]["acc_freq"]) > 0:
            metrics["acc_freq"] = self.metrics["freq"]["acc_freq"][-1]
            metrics["hamming_freq"] = self.metrics["freq"]["hamming_freq"][-1]
        return metrics

    def update_downstream_metrics_and_matrices(
        self, labels: np.ndarray, predictions: np.ndarray
    ):
        """
        Update the downstream metrics and matrices, this includes the precision, recall, f1-score, cohen kappa, confusion
        matrix and markov matrix.

        :param labels: the ground truth sleep stage labels
        :param predictions: the predicted sleep stage labels
        """
        self.update_downstream_metrics(labels, predictions)
        self.update_confusion_matrix(labels, predictions)
        self.update_markov_matrix(labels, predicted=False)
        self.update_markov_matrix(predictions, predicted=True)

    def update_downstream_loss(self, loss: float):
        """update the loss of the downstream model"""
        self.metrics["downstream"]["loss"].append(loss)

    def update_downstream_metrics(self, labels: np.ndarray, predictions: np.ndarray):
        """
        Update the precision, recall, f1-score and cohen kappa of the downstream model

        :param labels: the ground truth sleep stage labels
        :param predictions: the predicted sleep stage labels
        """
        stage_keys = list(self.stages.keys())
        sleep_stages = np.intersect1d(stage_keys, np.union1d(labels, predictions))
        # micro metrics per stage
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None, labels=sleep_stages, zero_division=0
        )
        for key, p, r, f in zip(sleep_stages.tolist(), prec, rec, f1):
            self.metrics["downstream"]["precisions"][self.stages[key]].append(p)
            self.metrics["downstream"]["recalls"][self.stages[key]].append(r)
            self.metrics["downstream"]["f1_scores"][self.stages[key]].append(f)
        # macro metrics (averaged)
        m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", labels=sleep_stages, zero_division=0
        )
        self.metrics["downstream"]["precisions"][KEY_MACRO].append(m_prec)
        self.metrics["downstream"]["recalls"][KEY_MACRO].append(m_rec)
        self.metrics["downstream"]["f1_scores"][KEY_MACRO].append(m_f1)

        self.metrics["downstream"]["cohen_kappa"].append(
            cohen_kappa_score(labels, predictions, labels=sleep_stages)
        )

    def get_last_macro_f1_score(self):
        """return the last macro f1-score"""
        return self.metrics["downstream"]["f1_scores"][KEY_MACRO][-1]

    def update_markov_matrix(self, labels: np.ndarray, predicted: bool = False):
        """
        Update the markov matrix of the sleep stage labels. The Markov matrix is only calculable if the labels are
        ordered.

        :param labels: the sleep stage labels
        :param predicted: if the labels are the predicted labels or the ground truth labels
        """
        n_stages = len(self.stages)
        if predicted:
            key = "predicted"
            self.markov_matrix[key].append(np.zeros((n_stages, n_stages), dtype="int"))
        else:
            # always the same so we only save one of them
            key = "actual"
            if len(self.markov_matrix[key]) == 0:
                self.markov_matrix[key].append(None)
            self.markov_matrix[key][0] = np.zeros((n_stages, n_stages), dtype="int")
        for i, stage in enumerate(labels):
            if i == 0:
                continue
            self.markov_matrix[key][-1][labels[i - 1], stage] += 1

    def update_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray):
        """update the confusion matrix of the sleep stage labels"""
        self.confusion_matrix.append(
            confusion_matrix(
                labels, predictions, labels=sorted(list(self.stages.keys()))
            )
        )

    def create_dict(self) -> dict:
        """
        Create a dictionary of the metrics and matrices that are used to evaluate the model.
        All empty lists and sub-dictionaries are removed from the final dictionary.
        """
        all_metrics = copy.deepcopy(
            {
                "metrics": self.metrics,
                "confusion_matrices": {
                    "abs": self.confusion_matrix,
                },
                "markov_matrices": {
                    "abs": self.markov_matrix,
                },
            }
        )
        all_metrics = remove_empty_values(all_metrics)

        return all_metrics
