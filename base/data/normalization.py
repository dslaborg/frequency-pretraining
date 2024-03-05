import numpy as np


def zscore(x: np.ndarray, axis: int | list[int]) -> np.ndarray:
    """
    Z-score normalize the input data. Standard deviations close to zero are replaced with ones to avoid division by
    zero.

    :param x: input data
    :param axis: axis to normalize over

    :return: z-score normalized data
    """
    x_mean = np.mean(x, axis=axis, keepdims=True)
    x_std = np.std(x, axis=axis, keepdims=True)
    # replace standard deviations close to zero with ones
    x_std = np.where(x_std > 1e-6, x_std, 1)
    return (x - x_mean) / x_std


def normalize_epoch(x: np.ndarray) -> np.ndarray:
    """
    Normalize an epoch by z-scoring each epoch.

    :param x: input data, datapoints are the last dimension

    :return: z-score normalized epoch
    """
    # normalize over last dimension, should be the datapoints, as the shapes are (epoch, channel, datapoints)
    return zscore(x, axis=-1)
