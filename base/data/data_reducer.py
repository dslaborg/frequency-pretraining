import numpy as np

from base.config import Config


class BaseDataReducer:
    def __init__(self, repeat_samples: bool, seed: int):
        self.repeat_samples = repeat_samples
        self.random_state = np.random.RandomState(seed)

    def reduce_data(
        self, x_indices: list[tuple[str, int, int]], y_data: dict[str, np.ndarray]
    ) -> list[tuple[str, int, int]]:
        """
        :param x_indices: List of tuples of the form (s_id, idx_start, idx_end) where s_id is the subject id, idx_start
        is the start index of the sequence of epochs, and idx_end is the end index of the sequence of epochs.
        :param y_data: Dictionary with the subject id as key and the corresponding sequence of stages as value.

        :return: Reduced list of tuples of the form (s_id, idx_start, idx_end).
        """
        raise NotImplementedError


class ClassWiseDataReducer(BaseDataReducer):
    def __init__(
        self,
        data_fraction: float,
        repeat_samples: bool,
        left_epochs: int,
        seed: int = None,
    ):
        """
        Reduces the data to a given fraction for each class.

        :param data_fraction: Fraction of the data to keep for each class.
        :param repeat_samples: If True, the samples will be repeated to have the same amount of samples regardless of
        data_fraction. This is useful for keeping the same number of gradient updates per training epoch.
        :param left_epochs: Number of epochs before the epoch of interest in the input sequences (samples). Needed
        because data reduction is based on the indices array from the dataloader, which has the following format:
        [(s_id, idx_start, idx_end), ...].
        :param seed: Random seed for reproducibility.
        """
        super().__init__(repeat_samples, seed)
        self.data_fraction = data_fraction
        self.left_epochs = left_epochs

    def reduce_data(
        self, x_indices: list[tuple[str, int, int]], y_data: dict[str, np.ndarray]
    ) -> list[tuple[str, int, int]]:
        """
        :param x_indices: List of tuples of the form (s_id, idx_start, idx_end) where s_id is the subject id, idx_start
        is the start index of the sequence of epochs, and idx_end is the end index of the sequence of epochs.
        :param y_data: Dictionary with the subject id as key and the corresponding sequence of stages as value.

        :return: Reduced list of tuples of the form (s_id, idx_start, idx_end).
        """
        # if data_fraction is 1.0, we don't need to do anything
        # also, if possible, we don't want to filter out artefacts to keep the signals continuous
        if self.data_fraction == 1.0:
            return x_indices
        _cfg = Config.get()
        stages = list(_cfg.data.stages.keys())

        # map the indices to the corresponding stages of the middle epoch in each sequence
        idx_per_stage = {stage: [] for stage in stages}  # implicitly excludes artifacts
        for s_id, idx_start, idx_end in x_indices:
            stage = y_data[s_id][idx_start + self.left_epochs]
            if stage in stages:
                idx_per_stage[stage].append((s_id, idx_start, idx_end))

        # reduce the number of samples for each stage
        for stage in idx_per_stage:
            n_samples = int(self.data_fraction * len(idx_per_stage[stage]))
            n_samples = max(1, n_samples)
            self.random_state.shuffle(idx_per_stage[stage])
            idx_per_stage[stage] = idx_per_stage[stage][:n_samples]

        x_indices = sorted(
            [elem for sublist in idx_per_stage.values() for elem in sublist]
        )

        if self.repeat_samples:
            # repeat all samples to have the same amount of samples regardless of data_fraction
            x_indices = x_indices * int(1 / self.data_fraction)

        return x_indices


class SubjectWiseFixedEpochsDataReducer(BaseDataReducer):
    def __init__(
        self,
        n_subjects: int,
        n_epochs: int,
        repeat_samples: bool,
        seed: int = None,
        **kwargs,
    ):
        """
        Reduces the data to a given number of subjects and epochs.

        :param n_subjects: Number of subjects to keep. If -1, all subjects are kept.
        :param n_epochs: Number of epochs to keep. If -1, all epochs are kept.
        :param repeat_samples: If True, the samples will be repeated to have the same amount of samples regardless of
        data_fraction. This is useful for keeping the same number of gradient updates per training epoch.
        :param seed: Random seed for reproducibility.
        """
        super().__init__(repeat_samples, seed)
        self.n_subjects = n_subjects
        self.n_epochs = n_epochs

    def reduce_data(
        self, x_indices: list[tuple[str, int, int]], y_data: dict[str, np.ndarray]
    ) -> list[tuple[str, int, int]]:
        """
        :param x_indices: List of tuples of the form (s_id, idx_start, idx_end) where s_id is the subject id, idx_start
        is the start index of the sequence of epochs, and idx_end is the end index of the sequence of epochs.
        :param y_data: Dictionary with the subject id as key and the corresponding sequence of stages as value.

        :return: Reduced list of tuples of the form (s_id, idx_start, idx_end).
        """
        # if n_subjects is -1 and n_epochs is -1, we don't need to do anything
        # also, if possible, we don't want to filter out artefacts to keep the signals continuous
        if (
            self.n_subjects == len(y_data.keys()) or self.n_subjects == -1
        ) and self.n_epochs == -1:
            return x_indices

        # sample n_subjects subjects
        s_ids = list(y_data.keys())
        self.random_state.shuffle(s_ids)
        if self.n_subjects != -1:
            s_ids = s_ids[: self.n_subjects]

        # filter the indices to keep only the selected subjects
        all_idx = [
            (s_id, idx_start, idx_end)
            for s_id, idx_start, idx_end in x_indices
            if s_id in s_ids
        ]

        # sample n_epochs epochs
        self.random_state.shuffle(all_idx)
        x_indices_red = sorted(
            all_idx[: self.n_epochs] if self.n_epochs != -1 else all_idx
        )

        if self.repeat_samples:
            # repeat all samples to have the same amount of samples regardless of data_fraction
            implicit_data_fraction = len(x_indices_red) / len(x_indices)
            x_indices_red = x_indices_red * int(1 / implicit_data_fraction)

        return x_indices_red


def get_class_distribution(
    x_indices: list[tuple[str, int, int]],
    y_data: dict[str, np.ndarray],
    left_epochs: int,
    for_print: bool = True,
) -> str | tuple[dict[str, int], dict[str, int]]:
    """
    Returns the distribution of samples per subject and stage.

    :param x_indices: List of tuples of the form (s_id, idx_start, idx_end) where s_id is the subject id, idx_start
    is the start index of the sequence of epochs, and idx_end is the end index of the sequence of epochs.
    :param y_data: Dictionary with the subject id as key and the corresponding sequence of stages as value.
    :param left_epochs: Number of epochs before the epoch of interest in the input sequences (samples).
    :param for_print: If True, the distribution is returned as a string. If False, the distribution is returned as a
    dictionary.

    :return: Distribution of samples per subject and stage. Either as a string or as two dictionaries.
    """
    y_filtered = np.array(
        [
            [s_id, y_data[s_id][idx_start + left_epochs]]
            for s_id, idx_start, idx_end in x_indices
        ]
    )
    dist_subjects = dict(zip(*np.unique(y_filtered[:, 0], return_counts=True)))
    dist_stages = dict(zip(*np.unique(y_filtered[:, 1], return_counts=True)))
    if for_print:
        str_rep = "# samples per subject"
        for s_id in dist_subjects:
            str_rep += f"\n{s_id}: {dist_subjects[s_id]}"
        str_rep += "\n\n# samples per stage"
        for stage in dist_stages:
            str_rep += f"\n{stage}: {dist_stages[stage]}"
        return str_rep
    else:
        return dist_subjects, dist_stages
