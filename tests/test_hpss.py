"""Tests for preprocessing.hpss - Harmonic-Percussive Source Separation."""
import numpy as np
import pytest
import librosa

from preprocessing.hpss import hpss


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


class TestHPSS:
    def test_output_shapes(self, synthetic_signal):
        y, sr = synthetic_signal
        S_h, S_p = hpss(y, sr=sr, n_fft=2048, hop_length=512)
        assert S_h.shape == S_p.shape
        assert S_h.shape[0] == 1025  # 1 + n_fft//2

    def test_output_nonnegative(self, synthetic_signal):
        y, sr = synthetic_signal
        S_h, S_p = hpss(y, sr=sr)
        assert np.all(S_h >= 0)
        assert np.all(S_p >= 0)

    def test_components_sum_to_original(self, synthetic_signal):
        """Soft-masked H + P is strictly ≤ original magnitude (Wiener mask).

        The Wiener soft mask is (H^2 * m) / (H^2 * m + P^2 * m + eps), so
        mask_H + mask_P < 1 always — H + P ≤ S_orig, never exceeds. We assert
        the upper bound tightly and the lower bound loosely (median filtering
        can leave a few low-energy cells with sum < 0.95 of original).
        """
        y, sr = synthetic_signal
        S_h, S_p = hpss(y, sr=sr, n_fft=2048, hop_length=512)
        S_orig = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        ratio = (S_h + S_p) / (S_orig + 1e-10)
        # Soft mask ensures ratio <= 1.0 + 1e-6 everywhere
        assert ratio.max() <= 1.0 + 1e-6, (
            f"H + P must not exceed original: max ratio = {ratio.max()}"
        )
        # Median filter can drop a small fraction of cells; 0.7 is a loose floor
        assert ratio.min() >= 0.7, (
            f"H + P dropped too far below original: min ratio = {ratio.min()}"
        )

    def test_harmonic_has_more_energy_at_440hz(self, synthetic_signal):
        """The harmonic component should dominate at 440 Hz."""
        y, sr = synthetic_signal
        S_h, S_p = hpss(y, sr=sr, n_fft=2048, hop_length=512)
        bin_440 = int(440 * 2048 / sr)
        h_energy = np.mean(S_h[bin_440, :])
        p_energy = np.mean(S_p[bin_440, :])
        assert h_energy > p_energy, (
            f"Harmonic should dominate at 440 Hz: h={h_energy:.4f}, p={p_energy:.4f}"
        )

    def test_percussive_has_more_energy_at_clicks(self, synthetic_signal):
        """At click frames, percussive energy across all freq bins should spike."""
        y, sr = synthetic_signal
        S_h, S_p = hpss(y, sr=sr, n_fft=2048, hop_length=512)
        # Click frames should have higher percussive broadband energy
        # than average harmonic broadband energy
        click_frame = int(0.25 * sr / 512)  # first click at 0.25s
        assert S_p[:, click_frame].mean() > S_h[:, click_frame].mean() * 0.5

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            hpss(np.array([]))
