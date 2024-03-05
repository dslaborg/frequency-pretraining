import time
from logging import getLogger

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.model.base_model import BaseModel
from base.results.result_tracker import ResultTracker

logger = getLogger(__name__)


class ClasTrainer:
    def __init__(
        self,
        epochs: int,
        model: BaseModel,
        dataloader: DataLoader,
        log_interval: int,
        clip_gradients: float | bool,
        lr: float | list[float],
        evaluators: dict[str, BaseEvaluator],
        early_stopping_epochs: int,
        seed: int = None,
    ):
        """
        Trainer for the downstream classification task (fine-tuning). Implements techniques like early stopping,
        gradient clipping, and learning rate scheduling.

        :param epochs: number of epochs to train
        :param model: model to train
        :param dataloader: dataloader with the training data
        :param log_interval: frequency of logs (every log_interval% of data in dataloader)
        :param clip_gradients: clip gradients to avoid exploding gradients, if False no clipping is done
        :param lr: learning rate for the optimizer, if list, then different learning rates are applied to the feature
            extractor and the classifier
        :param evaluators: dict of evaluators for the downstream task that are evaluated after each training epoch
        :param early_stopping_epochs: patience for early stopping
        :param seed: seed for reproducibility
        """
        self.epochs = epochs
        self.model = model
        self.dataloader = dataloader
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.lr = lr
        self.evaluators = evaluators
        self.early_stopping_epochs = early_stopping_epochs

        _cfg = Config.get()

        self.model.to(_cfg.general.device)
        # optimizer must be created after the model is initialized
        self.optimizer = instantiate(
            _cfg.training.downstream.optimizer,
            _convert_="all",  # needed because the model parameters are in a native dict and not a DictConfig
            params=self.model.get_parameters(self.lr),
        )
        # learning rate scheduler must be created after the optimizer is initialized
        self.lr_scheduler = instantiate(
            _cfg.training.downstream.lr_scheduler, optimizer=self.optimizer
        )

        # frequency of logs (every LOG_INTERVAL% of data in dataloader)
        self.log_fr = max(int(self.log_interval / 100.0 * len(self.dataloader)), 1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self) -> float:
        """training loop for the downstream task, evaluates the model after each epoch and logs the results"""
        # save best results for early stopping and snapshot creation of best model
        best_epoch = 0
        best_macro_f1_score = 0
        # log initial metrics before training
        logger.info(f"metrics before training (epoch 0):")
        for evaluator in self.evaluators.values():
            evaluator.evaluate(self.model)
            evaluator.log()

        for epoch in range(1, self.epochs + 1):
            start = time.time()  # measure time each epoch takes

            self.train_epoch(epoch)
            self.lr_scheduler.step()
            for evaluator in self.evaluators.values():
                evaluator.evaluate(self.model)

            end = time.time()
            logger.info(
                f"[epoch {epoch:3d}] execution time: {end - start:.2f}s\tmetrics:"
            )
            for evaluator in self.evaluators.values():
                evaluator.log()

            f1_score_es = ResultTracker.get("earlystopping").get_last_macro_f1_score()
            if f1_score_es > best_macro_f1_score:
                best_macro_f1_score = f1_score_es
                best_epoch = epoch
                self.model.save()

            # early stopping, stop training if the f1 score on the early stopping data has not increased
            # over the last x epochs
            if epoch - best_epoch >= self.early_stopping_epochs:
                break

        # write all results to a result file
        ResultTracker.write_all()
        logger.info("finished training")
        logger.info(
            f"best model on epoch: {best_epoch} \tf1-score: {best_macro_f1_score:.4f}"
        )

        return best_macro_f1_score

    def train_epoch(self, epoch: int):
        """training loop for one epoch"""
        _cfg = Config.get()
        train_res_tracker: ResultTracker = ResultTracker.get("train")

        self.model.train()

        # save predicted and actual labels for results
        predicted_labels = []
        actual_labels = []
        loss: torch.Tensor = torch.zeros(1, device=_cfg.general.device)

        for i, data in enumerate(self.dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            features: torch.Tensor = data[0]
            labels: torch.Tensor = data[1]
            features = features.to(_cfg.general.device)
            labels = (
                labels[:, self.dataloader.dataset.left_epochs]
                .long()
                .flatten()
                .to(_cfg.general.device)
            )
            # skip batches with only one sample, because otherwise the BN layers do not work
            if features.shape[0] == 1:
                continue

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(features)
            outputs = outputs[:, self.dataloader.dataset.left_epochs, :]

            # if the batch only contains artefacts, we skip the batch
            if not torch.all(labels == -1):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(
                    outputs[labels != -1], labels[labels != -1]
                )  # ignore artefacts (-1)

                loss.backward()
                if self.clip_gradients:
                    # clip the gradients to avoid exploding gradients
                    if isinstance(self.clip_gradients, float):
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_gradients, "inf"
                        )
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, "inf")
                self.optimizer.step()

            # determine predicted labels and save them together with actual labels
            _, predicted_labels_i = torch.max(outputs, dim=1)
            predicted_labels.append(predicted_labels_i)
            actual_labels.append(labels)

            # log various information every log_fr minibatches
            if i % self.log_fr == self.log_fr - 1:
                logger.info(
                    f"train epoch: {epoch} [{i * len(features)}/{len(self.dataloader.dataset)} "
                    f"({100. * i / len(self.dataloader):.0f}%)], "
                    f'lr: {[f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr()]}, '
                    f"loss: {loss.item():.6f}"
                )

        predicted_labels = torch.cat(predicted_labels, dim=0).cpu().long().numpy()
        actual_labels = torch.cat(actual_labels, dim=0).cpu().long().numpy()

        # remove artefacts from the predicted and actual labels
        artefacts_mask = actual_labels != -1
        actual_labels = actual_labels[artefacts_mask]
        predicted_labels = predicted_labels[artefacts_mask]

        train_res_tracker.update_downstream_loss(loss.item())
        train_res_tracker.update_downstream_metrics_and_matrices(
            actual_labels, predicted_labels
        )
