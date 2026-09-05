"""Harmonic-Percussive Source Separation - wraps librosa.decompose.hpss."""
import numpy as np
import librosa

from preprocessing.stft import compute_stft


def hpss(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_size: int = 31,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Harmonic-Percussive Source Separation.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Audio time series.
    sr : int
        Sample rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length.
    kernel_size : int
        Median filter width for both harmonic and percussive.
    margin : float
        Margin for soft masking (>= 1.0).

    Returns
    -------
    S_harmonic : np.ndarray, shape (1 + n_fft//2, n_frames), float
        Magnitude spectrogram of harmonic component.
    S_percussive : np.ndarray, shape (1 + n_fft//2, n_frames), float
        Magnitude spectrogram of percussive component.

    Raises
    ------
    ValueError
        If input signal is empty.
    """
    S = compute_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_h, S_p = librosa.decompose.hpss(S, kernel_size=kernel_size, margin=margin)
    return np.abs(S_h), np.abs(S_p)
