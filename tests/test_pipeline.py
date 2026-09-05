"""Tests for preprocessing.pipeline: end-to-end STFT + HPSS."""

import numpy as np
import pytest

from preprocessing.pipeline import process_audio


@pytest.fixture
def synthetic_signal():
    """2-second synthetic signal: 440 Hz sine + 0.25s click impulses."""
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    harmonic = 0.5 * np.sin(2 * np.pi * 440 * t)
    percussive = np.zeros_like(t)
    for ct in np.arange(0, 2.0, 0.25):
        idx = int(ct * sr)
        percussive[idx : idx + int(0.005 * sr)] = 0.8
    return harmonic + percussive, sr


class TestProcessAudio:
    def test_returns_dict_with_required_keys(self, synthetic_signal):
        y, sr = synthetic_signal
        result = process_audio(y, sr=sr)
        assert isinstance(result, dict)
        assert "melodic" in result
        assert "percussive" in result

    def test_output_shapes_match(self, synthetic_signal):
        y, sr = synthetic_signal
        result = process_audio(y, sr=sr, n_fft=2048, hop_length=512)
        # HPSS uses compute_stft() with librosa default center=True, so
        # n_frames = 1 + len(y) // hop_length = 1 + 44100 // 512 = 87.
        assert result["melodic"].shape == result["percussive"].shape
        assert result["melodic"].shape == (1025, 87)

    def test_output_nonnegative(self, synthetic_signal):
        y, sr = synthetic_signal
        result = process_audio(y, sr=sr)
        assert np.all(result["melodic"] >= 0)
        assert np.all(result["percussive"] >= 0)
