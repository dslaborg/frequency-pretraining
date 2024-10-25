import numpy as np


class MetaBase:
    def load_data(
        self, subject_ids: list[str], channels: list[str]
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
        """
        :param subject_ids: list of subject ids to filter for
        :param channels: list of channels to load

        :return: x_data, y_data
        """
        raise NotImplementedError
