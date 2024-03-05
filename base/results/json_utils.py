import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return str(obj.tolist())
        return str(obj)


def remove_empty_values(dictionary: dict) -> dict:
    """Remove empty values from a dictionary recursively"""
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            remove_empty_values(value)
        if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
            dictionary.pop(key)
    return dictionary
