import numpy as np
import scipy.signal


class BaseDataAug:
    def __init__(self, seed=None):
        self.random_state = np.random.RandomState(seed)

    def augment(self, x):
        return x


class AmplitudeScale(BaseDataAug):
    def __init__(self, max_fac, seed=None):
        super().__init__(seed)
        self.max_fac = max_fac

    def augment(self, x):
        """scale the amplitude of the signal by a random factor between 1 and max_fac
        or between 1/max_fac and 1
        same factor for all channels"""
        x *= np.pow(
            (self.random_state.random_sample(1)) * (self.max_fac - 1) + 1,
            self.random_state.randint(0, 2) * 2 - 1,
        )
        return x


class DCShift(BaseDataAug):
    def __init__(self, max_shift, seed=None):
        super().__init__(seed)
        self.max_shift = max_shift

    def augment(self, x):
        """shift the amplitude by a random amount between -max_shift and max_shift
        same shift for all channels"""
        x += self.random_state.random_sample(1) * 2 * self.max_shift - self.max_shift
        return x


class NormalNoise(BaseDataAug):
    def __init__(self, sigma, seed=None):
        super().__init__(seed)
        self.sigma = sigma

    def augment(self, x):
        """add normally distributed noise to the signal
        different noise for each channel"""
        x += self.random_state.randn(*x.shape) * self.sigma
        return x


class SegmentMask(BaseDataAug):
    def __init__(self, lower, upper, num, same_over_channels=False, seed=None):
        super().__init__(seed)
        self.lower = lower
        self.upper = upper
        self.num = num
        self.same_over_channels = same_over_channels

    def augment(self, x):
        """mask a segment of the signal"""
        idx_mask = np.arange(x.shape[1])[None, :].repeat(x.shape[0], axis=0)

        for _ in range(self.num):
            width = (
                np.random.rand(x.shape[0], 1) * (self.upper - self.lower) + self.lower
            ) * x.shape[-1]
            start = np.random.rand(x.shape[0], 1) * (x.shape[-1] - width)

            if self.same_over_channels:
                width[1:, 0] = width[0, 0]
                start[1:, 0] = start[0, 0]

            x[(idx_mask > start) & (idx_mask < (start + width))] = 0

        return x


class TimeShift(BaseDataAug):
    def __init__(self, max_shift, seed=None):
        super().__init__(seed)
        self.max_shift = max_shift

    def augment(self, x):
        """shift the signal temporally by a random amount between -max_shift and max_shift
        same shift for each channel"""
        shift = np.random.random_sample() * 2 * self.max_shift - self.max_shift
        shift = int(shift * x.shape[-1])

        x = np.roll(x, shift, axis=-1)
        if shift > 0:
            x[:, :shift] = 0
        else:
            x[:, shift:] = 0

        return x


class TimeWarp(BaseDataAug):
    def __init__(self, max_fac, seed=None):
        super().__init__(seed)
        self.max_fac = max_fac

    def augment(self, x):
        """warp the signal temporally by a random factor between 1 and max_fac
        or between 1/max_fac and 1
        same factor for each channel"""
        new_size = int(
            np.pow(
                (np.random.rand()) * (self.max_fac - 1) + 1,
                np.random.randint(0, 2) * 2 - 1,
            )
            * x.shape[-1]
        )

        start = (x.shape[-1] - new_size) // 2
        end = start + new_size

        x_re = scipy.signal.resample_poly(x, new_size, x.shape[-1], axis=-1)

        x_tmp = np.zeros(x.shape)
        x_tmp[:, max(0, start) : min(x_tmp.shape[-1], end)] = x_re[
            :, max(0, -start) : min(new_size, x_tmp.shape[-1] - start)
        ]

        return x_tmp


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # x = np.random.random_sample((2, 3000)) - 0.5
    x = np.sin(np.linspace(0, 10 * np.pi, 3000))[None, :]
    aug = TimeWarp(0.5)
    x_aug = aug.augment(x)
    _, axs = plt.subplots(2, 1)
    axs[0].plot(x[0])
    axs[1].plot(x_aug[0])
    plt.show()
