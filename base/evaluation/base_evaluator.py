from torch import nn
from torch.utils.data import DataLoader


class BaseEvaluator:
    def __init__(self, dataloader: DataLoader, dataset: str):
        """
        :param dataloader: The dataloader to evaluate the model on.
        :param dataset: The name of the dataset to evaluate the model on. Used for logging and to distinguish between
        different result_tracker instances.
        """
        self.dataloader = dataloader
        self.dataset = dataset

    def evaluate(self, model: nn.Module) -> None:
        """Evaluate the model on the given dataset."""
        raise NotImplementedError

    def log(self):
        """Log the evaluation results to the console or a file."""
        raise NotImplementedError
