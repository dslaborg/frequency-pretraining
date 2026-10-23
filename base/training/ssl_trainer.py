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


class SSLTrainer:
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
        Trainer for generic SSL tasks

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
        """training loop"""
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
        loss_components = []

        for i, data in enumerate(self.dataloader, 0):
            # get the inputs; data is a list of [x1, x2]
            features_1, features_2 = data
            features_1 = features_1.to(_cfg.general.device)
            features_2 = features_2.to(_cfg.general.device)
            # skip batches with only one sample, because otherwise the BN layers do not work
            if features_1.shape[0] == 1:
                continue

            # zero the parameter gradients
            self.optimizer.zero_grad()

            loss_i, loss_components_i = self.model((features_1, features_2))

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

            loss += loss_i.detach() * features_1.size(0)
            if len(loss_components) == 0:
                loss_components = [torch.zeros(1)] * len(loss_components_i)
            loss_components = [
                l + li * features_1.size(0)
                for l, li in zip(loss_components, loss_components_i)
            ]

            # log various information every log_fr minibatches
            if i % self.log_fr == self.log_fr - 1:
                logger.info(
                    f"train epoch: {epoch} [{i * len(features_1)}/{len(self.dataloader.dataset)} "
                    f"({100. * i / len(self.dataloader):.0f}%)], "
                    f'lr: {[f"{lr:.2e}" for lr in self.lr_scheduler.get_last_lr()]}, '
                    f"loss: {loss_i.item():.6f}"
                )

        train_res_tracker.update_ssl_loss(loss.item() / len(self.dataloader.dataset))
        train_res_tracker.update_ssl_loss_components(
            [l.item() / len(self.dataloader.dataset) for l in loss_components]
        )
