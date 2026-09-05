"""End-to-end spectrogram profiling pipeline: audio in, melodic/percussive out."""

import numpy as np

from preprocessing.hpss import hpss


def process_audio(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_size: int = 31,
    margin: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Process an audio signal into melodic and percussive magnitude spectrograms.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,)
        Audio time series.
    sr : int
        Sample rate.
    n_fft, hop_length, kernel_size, margin
        Forwarded to HPSS.

    Returns
    -------
    dict with keys:
        "melodic"    : np.ndarray, shape (1 + n_fft//2, n_frames)
            Harmonic (sustained-pitch) magnitude spectrogram.
        "percussive" : np.ndarray, same shape
            Percussive (transient) magnitude spectrogram.
    """
    S_h, S_p = hpss(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        kernel_size=kernel_size,
        margin=margin,
    )
    return {"melodic": S_h, "percussive": S_p}
