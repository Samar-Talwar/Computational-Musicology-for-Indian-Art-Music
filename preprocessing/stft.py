"""STFT computation - wraps librosa.stft with project defaults."""

import numpy as np
import librosa


def compute_stft(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Audio time series.
    sr : int
        Sample rate (for documentation; not used in computation).
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length between frames.

    Returns
    -------
    np.ndarray, shape (1 + n_fft//2, n_frames), complex
        STFT matrix.

    Raises
    ------
    ValueError
        If input signal is empty.
    """
    if len(y) == 0:
        raise ValueError("Input signal is empty")
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def compute_magnitude_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute magnitude spectrogram (|STFT|).

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

    Returns
    -------
    np.ndarray, shape (1 + n_fft//2, n_frames), float
        Magnitude spectrogram.
    """
    return np.abs(compute_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length))
