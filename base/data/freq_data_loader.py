import numpy as np
from torch.utils.data import Dataset

from base.config import Config
from base.data.normalization import normalize_epoch


class DatasetForRandomFrequencies(Dataset):
    def __init__(
        self,
        n_samples: int = 1000,
        n_freq: int = 20,
        freq_min: float = 0.3,
        freq_max: float = 35,
        sr: int = 100,
        seconds: int = 30,
        do_phase_shifts: bool = False,
        normalize: bool = True,
        log_bins: bool = True,
        seed: int = None,
    ):
        """
        Dataset for the creation of synthetic time series data with random frequencies. Each sample consists of a
        synthetic time series with the same number of channels used during fine-tuning and the corresponding labels for
        each frequency bin. Synthetic signals have a length of seconds * sr.
        Samples are currently not cached, but generated on-the-fly during training. Nonetheless, each training epoch uses
        the same data.

        Data creation process:
          1. Divide the frequency range [freq_min, freq_max] into n_freq bins. Bins can be linearly or logarithmically
             spaced (log_bins).
          2. Randomly choose the number of frequencies for each sample from 1 to n_freq.
          3. Randomly choose the frequencies for each channel and each sample from within the frequency bins.
          4. Randomly choose phase shifts for each frequency bin and each sample if do_phase_shifts is True.
          5. Create the synthetic time series by summing up the sine waves for each frequency bin and each channel.
          6. Normalize the time series if normalize is True.

        :param n_samples: number of samples to generate
        :param n_freq: number of frequency bins
        :param freq_min: minimum frequency
        :param freq_max: maximum frequency
        :param sr: sampling rate
        :param seconds: length of the synthetic time series in seconds
        :param do_phase_shifts: if True, random phase shifts are added to the sine waves
        :param normalize: if True, normalize the synthetic time series
        :param log_bins: if True, use logarithmically spaced frequency bins
        :param seed: random seed for reproducibility
        """
        super(DatasetForRandomFrequencies, self).__init__()
        self.n_samples = n_samples
        self.n_freqs = n_freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.sr = sr
        self.seconds = seconds
        self.do_phase_shifts = do_phase_shifts
        self.normalize = normalize
        self.random_state = np.random.RandomState(seed)

        _cfg = Config.get()
        # get number of channels from the config of the downstream task (fine-tuning)
        self.n_channels = len(_cfg.data.downstream.train_dataloader.dataset.channels)

        # initialize data to generate so each training epoch uses the same data
        # randomly choose the number of frequencies for each sample
        number_of_freqs = self.random_state.randint(1, self.n_freqs + 1, self.n_samples)
        freq_bins = np.arange(self.n_freqs)
        self.chosen_freq_bins = [
            self.random_state.choice(freq_bins, number_of_freqs[s], replace=False)
            for s in range(self.n_samples)
        ]

        # randomly choose frequencies for each channel and each sample
        if log_bins:
            freq_ranges = np.logspace(
                np.log2(self.freq_min), np.log2(self.freq_max), self.n_freqs + 1, base=2
            )
        else:
            freq_ranges = np.linspace(self.freq_min, self.freq_max, self.n_freqs + 1)
        self.chosen_freqs = [
            [
                [
                    (freq_ranges[b + 1] - freq_ranges[b])
                    * self.random_state.random_sample(1)
                    + freq_ranges[b]
                    for b in self.chosen_freq_bins[s]
                ]
                for _ in range(self.n_channels)
            ]
            for s in range(self.n_samples)
        ]

        # randomly choose phase shifts for each frequency bin and each sample
        self.phase_shifts = [
            [
                (self.random_state.random_sample(1) * 2 * np.pi)
                if self.do_phase_shifts
                else 0
                for _ in self.chosen_freq_bins[s]
            ]
            for s in range(self.n_samples)
        ]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset. index determines which frequencies and phases are used to create the synthetic
        time series. The sample consists of the synthetic time series and the corresponding labels for each frequency
        bin.

        :return: tuple of the form (x, y) where x is a tensor of shape (n_channels, datapoints) and y is a tensor of
        shape (n_freqs).
        """
        t = np.linspace(0, 1, self.sr * self.seconds)
        x = np.array(
            [
                np.sum(
                    [
                        np.sin(2 * np.pi * t * self.seconds * f + ps)
                        for f, ps in zip(
                            self.chosen_freqs[index][ch], self.phase_shifts[index]
                        )
                    ],
                    axis=0,
                )
                for ch in range(self.n_channels)
            ]
        )

        if self.normalize:
            x = normalize_epoch(x, "zscore")

        # create one-hot encoded labels for the chosen frequency bins
        y = np.zeros(self.n_freqs)
        y[self.chosen_freq_bins[index]] = 1
        return np.array(x, dtype="float32"), y

    def __len__(self):
        return self.n_samples
