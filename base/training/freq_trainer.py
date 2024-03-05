import time
from logging import getLogger

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader

from base.config import Config
from base.evaluation.base_evaluator import BaseEvaluator
from base.results.result_tracker import ResultTracker

logger = getLogger(__name__)


class FreqTrainer:
    def __init__(
        self,
        epochs: int,
        model: nn.Module,
        dataloader: DataLoader,
        log_interval: int,
        clip_gradients: bool | float,
        evaluators: dict[str, BaseEvaluator],
        add_epoch_in_save: bool = False,
        seed: int = None,
    ):
        """
        Trainer for the frequency pretraining task

        :param epochs: number of epochs to train
        :param model: model to train
        :param dataloader: dataloader for the training data
        :param log_interval: frequency of logs (every log_interval% of data in dataloader)
        :param clip_gradients: whether to clip gradients to avoid exploding gradients, can be a float to specify the
            maximum gradient norm
        :param evaluators: dictionary of evaluators to evaluate the model after each training epoch
        :param add_epoch_in_save: whether to add the epoch number to the saved model name, if False the saved model is
            overwritten after each training epoch
        :param seed: random seed for reproducibility
        """
        self.epochs = epochs
        self.model = model
        self.dataloader = dataloader
        self.log_interval = log_interval
        self.clip_gradients = clip_gradients
        self.evaluators = evaluators
        self.add_epoch_in_save = add_epoch_in_save

        _cfg = Config.get()

        self.model.to(_cfg.general.device)
        # optimizer can only be instantiated after the model is initialized
        self.optimizer = instantiate(
            _cfg.training.pretraining.optimizer,
            _convert_="all",
            params=[
                {"params": self.model.parameters()},
            ],
        )
        # lr_scheduler can only be instantiated after the optimizer is initialized
        self.lr_scheduler = instantiate(
            _cfg.training.pretraining.lr_scheduler, optimizer=self.optimizer
        )

        # frequency of logs (every LOG_INTERVAL% of data in dataloader)
        self.log_fr = max(int(self.log_interval / 100.0 * len(self.dataloader)), 1)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def train(self) -> float:
        """training loop for the frequency pretraining task"""
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

            # save model every epoch (we assume that the model does not get worse with more training)
            # if add_epoch_in_save is set to True, the epoch number is added to the saved model name
            # otherwise the saved model is overwritten after each training epoch
            self.model.save(epoch=epoch if self.add_epoch_in_save else None)

        # write all results to file
        ResultTracker.write_all()
        logger.info("finished training")

        return 0  # no metric for measuring the quality of representations

    def train_epoch(self, epoch: int):
        """training loop for one epoch"""
        _cfg = Config.get()
        train_res_tracker = ResultTracker.get("train")

        self.model.train()

        # aggregate statistics over the entire epoch
        loss = torch.zeros(1, device=_cfg.general.device)
        predicted_labels = []
        actual_labels = []

        for i, data in enumerate(self.dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            features, labels = data
            features = features.to(_cfg.general.device)
            labels = labels.float().to(_cfg.general.device)
            # skip batches with only one sample, because otherwise the BN layers do not work
            if features.shape[0] == 1:
                continue

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(features)
            criterion = nn.BCEWithLogitsLoss()
            loss_i = criterion(outputs, labels)

            loss_i.backward()
            if self.clip_gradients:
                # clip the gradients to avoid exploding gradients
                if isinstance(self.clip_gradients, float):
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_gradients, "inf"
                    )
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.1, "inf")
            self.optimizer.step()

            loss += loss_i.detach() * features.size(0)

            predicted_labels_i = torch.sigmoid(outputs.detach()) > 0.5
            predicted_labels.append(predicted_labels_i)
            actual_labels.append(labels)

            # log various information every log_fr minibatches
            if i % self.log_fr == self.log_fr - 1:
                logger.info(
                    f"train epoch: {epoch} [{i * len(features)}/{len(self.dataloader.dataset)} "
                    f"({100. * i / len(self.dataloader):.0f}%)], "
                    f'lr: {[f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr()]}, '
                    f"loss: {loss_i.item():.6f}"
                )

        predicted_labels = torch.cat(predicted_labels, dim=0).cpu().long().numpy()
        actual_labels = torch.cat(actual_labels, dim=0).cpu().long().numpy()
        train_res_tracker.update_freq_loss(loss.item() / len(self.dataloader.dataset))
        train_res_tracker.update_freq_metrics(actual_labels, predicted_labels)
