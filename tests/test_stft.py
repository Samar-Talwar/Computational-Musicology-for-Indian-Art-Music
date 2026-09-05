"""Tests for preprocessing.stft - STFT computation."""

import numpy as np
import pytest
import librosa

from preprocessing.stft import compute_stft, compute_magnitude_spectrogram


@pytest.fixture
def synthetic_signal():
    """2-second synthetic signal: 440 Hz sine + clicks."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    harmonic = 0.5 * np.sin(2 * np.pi * 440 * t)
    percussive = np.zeros_like(t)
    for ct in np.arange(0, 2.0, 0.25):
        idx = int(ct * sr)
        percussive[idx : idx + int(0.005 * sr)] = 0.8
    return harmonic + percussive, sr


class TestComputeSTFT:
    def test_output_shape(self, synthetic_signal):
        y, sr = synthetic_signal
        S = compute_stft(y, sr=sr, n_fft=2048, hop_length=512)
        # shape: (1 + n_fft//2, n_frames)
        assert S.shape[0] == 1025
        expected_frames = 1 + len(y) // 512  # For center=True (librosa default)
        assert S.shape[1] == expected_frames  # Exact match, no tolerance needed

    def test_output_is_complex(self, synthetic_signal):
        y, sr = synthetic_signal
        S = compute_stft(y, sr=sr)
        assert np.iscomplexobj(S)

    def test_matches_librosa(self, synthetic_signal):
        y, sr = synthetic_signal
        S_ours = compute_stft(y, sr=sr, n_fft=2048, hop_length=512)
        S_ref = librosa.stft(y, n_fft=2048, hop_length=512)
        min_frames = min(S_ours.shape[1], S_ref.shape[1])
        max_diff = np.max(np.abs(S_ours[:, :min_frames] - S_ref[:, :min_frames]))
        assert max_diff < 1e-6, f"STFT diverges from librosa: max diff {max_diff}"

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_stft(np.array([]))


class TestComputeMagnitudeSpectrogram:
    def test_output_shape(self, synthetic_signal):
        y, sr = synthetic_signal
        mag = compute_magnitude_spectrogram(y, sr=sr, n_fft=2048, hop_length=512)
        assert mag.shape[0] == 1025

    def test_output_nonnegative(self, synthetic_signal):
        y, sr = synthetic_signal
        mag = compute_magnitude_spectrogram(y, sr=sr)
        assert np.all(mag >= 0)

    def test_matches_abs_of_stft(self, synthetic_signal):
        y, sr = synthetic_signal
        S = compute_stft(y, sr=sr)
        mag = compute_magnitude_spectrogram(y, sr=sr)
        np.testing.assert_allclose(mag, np.abs(S))
