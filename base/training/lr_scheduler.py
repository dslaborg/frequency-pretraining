# credits to https://github.com/abhuse/cyclic-cosine-decay
from collections.abc import Iterable
from math import log, cos, pi, floor

from torch.optim.lr_scheduler import LRScheduler


class CyclicCosineDecayLR(LRScheduler):
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr_multiplier,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr_multiplier=None,
                 warmup_epochs=None,
                 warmup_start_lr_multiplier=None,
                 last_epoch=-1):
        """
        Initialize new CyclicCosineDecayLR object.

        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr_multiplier: (float or iterable of floats) - learning rate of optimizer * min_decay_lr_multiplier
            is learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr_multiplier: (float or iterable of floats) - learning rate of optimizer * restart_lr_multiplier
            is learning rate when cycle restarts. If None, 1.0 will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr_multiplier: (float or iterable of floats) - learning rate of optimizer * warmup_start_lr_multiplier
            is learning rate at the beginning of warmup. Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr_multiplier, Iterable) and len(min_decay_lr_multiplier) != len(
                optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr_multiplier) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr_multiplier), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier,
                                float) or restart_interval_multiplier <= 0 or restart_interval_multiplier == 1):
            raise ValueError(
                "restart_interval_multiplier must be positive float and must not be 1, got {} instead".format(
                    restart_interval_multiplier))

        if isinstance(restart_lr_multiplier, Iterable) and len(restart_lr_multiplier) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr_multiplier) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr_multiplier), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr_multiplier is None:
                raise ValueError("warmup_start_lr_multiplier must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr_multiplier, float) or isinstance(warmup_start_lr_multiplier, Iterable)):
                raise ValueError(
                    "warmup_start_lr_multiplier must be either float or iterable of floats, got {} instead".format(
                        warmup_start_lr_multiplier))

            if isinstance(warmup_start_lr_multiplier, Iterable) and len(warmup_start_lr_multiplier) != len(
                    optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr_multiplier) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr_multiplier),
                                                                len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        if warmup_start_lr_multiplier is None:
            self._warmup_start_lr = None
        elif isinstance(warmup_start_lr_multiplier, float):
            self._warmup_start_lr = [warmup_start_lr_multiplier * base_lrs[i] for i in range(group_num)]
        else:
            self._warmup_start_lr = [warmup_start_lr_multiplier[i] * base_lrs[i] for i in range(group_num)]
        if isinstance(min_decay_lr_multiplier, float):
            self._min_decay_lr = [min_decay_lr_multiplier * base_lrs[i] for i in range(group_num)]
        else:
            self._min_decay_lr = [min_decay_lr_multiplier[i] * base_lrs[i] for i in range(group_num)]
        if restart_lr_multiplier is None:
            self._restart_lr = None
        elif isinstance(restart_lr_multiplier, float):
            self._restart_lr = [restart_lr_multiplier * base_lrs[i] for i in range(group_num)]
        else:
            self._restart_lr = [restart_lr_multiplier[i] * base_lrs[i] for i in range(group_num)]
        super(CyclicCosineDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            # return self._calc(self.last_epoch,
            #                   self._warmup_epochs,
            #                   self._warmup_start_lr,
            #                   self.base_lrs)
            # linear warmup
            return [wu_lr + (base_lr - wu_lr) * self.last_epoch / self._warmup_epochs
                    for wu_lr, base_lr in zip(self._warmup_start_lr, self.base_lrs)]

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) \
                                  % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                1 - self._restart_interval_multiplier)
