from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.signal import butter as _butter, resample_poly
from scipy.signal import filtfilt as _filtfilt
from scipy.signal import sosfiltfilt as _sosfiltfilt


def _highpass(sr: float, f: float, outtype="sos") -> tuple[np.ndarray, np.ndarray]:
    highpass_freq = 2 * f / sr
    return _butter(4, highpass_freq, btype="high", output=outtype)


def _lowpass(sr: float, f: float, outtype="sos") -> tuple[np.ndarray, np.ndarray]:
    lowpass_freq = 2 * f / sr
    return _butter(4, lowpass_freq, btype="low", output=outtype)


def _bandpass(
    sr: float, band: tuple[float, float], outtype="sos"
) -> tuple[np.ndarray, np.ndarray]:
    bandpass_freqs = 2 * np.array(band) / sr
    return _butter(4, bandpass_freqs, btype="band", output=outtype)


@lru_cache(maxsize=128)
def _pass(
    sr: float, band: tuple[float, float], outtype="sos"
) -> tuple[np.ndarray, np.ndarray]:
    try:
        assert not all(f is None for f in band), "fmin and fmax is `None`."
        if band[1] is None:
            return _highpass(sr, band[0], outtype)
        elif band[0] is None:
            return _lowpass(sr, band[1], outtype)
        else:
            assert band[0] < band[1], "fmin>=fmax (fmin: %s, fmax: %s)" % band
            return _bandpass(sr, band, outtype)
    except AssertionError as err:
        raise ValueError(str(err))


def filtfilt(x, sr, fmin=None, fmax=None, axis=-1, outtype="sos", method="pad"):
    """applies filter based on `outtype`"""
    if outtype == "sos":
        sos = _pass(sr, (fmin, fmax), outtype)
        return _sosfiltfilt(sos, x, axis=axis, padtype="constant")
    elif outtype == "ba":
        b, a = _pass(sr, (fmin, fmax), outtype)
        return _filtfilt(b, a, x, axis=axis, padtype="constant", method=method)
    else:
        raise ValueError("outtype neither sos nor ba")


def downsample(x, sr_old, sr_new, fmin=None, fmax=None, outtype="sos", method="pad"):
    """
    Applies 8-th order Butterworth filter with to `x`. Depending on whether `fmin`
    and `fmax` are `None`, a high-pass, low-pass or band-pass filter is applied.
    Afterwards, the signal is down-sampled to `sr_new`. If `sr_new==sr_old`, the
    filter is still applied, but the signal is not down-sampled. Down-sampling is performed
    using a polyphase filter from scipy.signal.resample_poly.

    Args:
        x (np.ndarray): samples to be down-sampled.
        sr_old (float): sampling rate in Hz of `x`.
        sr_new (float): sampling rate in Hz to which `x` shall be resampled.
        fmin (float or None): high-pass edge of the filter to be applied to `x` before down-sampling.
        fmax (float or None): low-pass edge of the filter to be applied to `x` before down-sampling.
        outtype (str): 'sos' or 'ba', type of filter to be used, scipy.signal.filtfilt or scipy.signal.sosfiltfilt
        method (str): 'pad' or 'gust', only used if method is 'ba', for more details see scipy.signal.filtfilt

    Returns:
        np.ndarray: down-sampled signal
    """
    # Filter before down-sampling
    try:
        assert fmax <= 0.4 * sr_new, (
            """fmax %s > 0.8*f_nyquist of new sampling rate""" % fmax
        )
    except AssertionError as err:
        raise ValueError(str(err))
    x = filtfilt(x, sr=sr_old, fmin=fmin, fmax=fmax, outtype=outtype, method=method)

    # Resample to new sampling rate
    if not sr_old == sr_new:
        x = resample_poly(x, int(sr_new), int(sr_old))
    return x


if __name__ == "__main__":
    # Example

    import matplotlib.pyplot as plt

    sr_old = 200
    sr_new = 400
    t = np.linspace(0, 2, 2 * sr_old)
    x = (
        np.sin(2 * np.pi * 10 * t)  # 10 Hz
        + np.sin(2 * np.pi * 30 * t)  # 30 Hz
        + np.sin(2 * np.pi * 50 * t)  # 50 Hz
        + t  # linear trend
    )
    x_new = downsample(x, sr_old, sr_new, fmin=0.2, fmax=32)
    t_new = np.linspace(0, 2, len(x_new))
    plt.plot(t, x, label="original")
    plt.plot(t_new, x_new, label="downsampled")
    plt.legend()

    # plot power spectrum
    plt.figure()
    f, Pxx_den = signal.periodogram(x, sr_old)
    f_new, Pxx_den_new = signal.periodogram(x_new, sr_new)
    plt.semilogy(f, Pxx_den, label="original")
    plt.semilogy(f_new, Pxx_den_new, label="downsampled")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.ylim([10e-6, plt.ylim()[1]])
    plt.legend()
    plt.show()
