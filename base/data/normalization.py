import numpy as np
from sklearn.preprocessing import RobustScaler

from base.config import Config


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


def iqr_scale(x: np.ndarray, axis: int | list[int]) -> np.ndarray:
    """
    Scale the input data using the interquartile range. The data is first shifted to have a mean of zero and then
    scaled by the interquartile range. Outliers are clamped to the value specified in the config.

    :param x: input data
    :param axis: axis to normalize over

    :return: iqr scaled data
    """
    if type(axis) is int:
        axis = [axis]
    _cfg = Config.get()
    robust_scaler = RobustScaler()
    clamp_value = _cfg.data.clamp_value

    shape = np.array(x.shape)
    axis = np.where(np.array(axis) < 0, np.array(axis) + len(shape), axis)
    other_axes = np.array([i for i in range(len(shape)) if i not in axis])
    permute_axes = np.r_[axis, other_axes]
    x = x.transpose(permute_axes).reshape(np.prod([shape[i] for i in axis]), -1)
    x = robust_scaler.fit_transform(x)
    x[x < -clamp_value] = -clamp_value
    x[x > clamp_value] = clamp_value
    x = x.reshape(*shape[permute_axes]).transpose(np.argsort(permute_axes))
    return x


def normalize_epoch(x: np.ndarray, norm_type: str) -> np.ndarray:
    """
    Normalize an epoch by z-scoring each epoch.

    :param x: input data, datapoints are the last dimension
    :param norm_type: normalization type, currently only 'zscore' and 'iqr' are supported

    :return: normalized epoch(s)
    """
    if norm_type == "zscore":
        # normalize over last dimension, should be the datapoints, as the shapes are (epoch, channel, datapoints)
        return zscore(x, axis=-1)
    elif norm_type == "iqr":
        return iqr_scale(x, axis=-1)
    else:
        raise ValueError(f"Normalization type {norm_type} is not supported")
