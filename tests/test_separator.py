"""Test suite for models/separator.py"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add worktree root to path so `models` resolves
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_separate_stereo_preserves_both_channels():
    """Stereo input yields 4 stems, each stereo, both channels preserved."""
    from models.separator import separate

    # 1-second stereo at 44100 Hz, distinct channels
    audio = np.random.randn(2, 44100).astype(np.float32)
    audio[0, :] *= 2.0  # L channel louder

    stems = separate(audio, sr=44100)

    assert stems.shape == (4, 2, 44100), f"Expected (4,2,44100), got {stems.shape}"
    assert stems.dtype == np.float32
    # Verify channels are distinct (not duplicated)
    for i in range(4):
        assert not np.array_equal(stems[i, 0], stems[i, 1]), \
            f"Stem {i}: L and R channels are identical (silent duplication bug)"


def test_separate_mono_duplicates_to_stereo():
    """Mono input (1D array) is accepted, duplicated to stereo for inference, and 4 stereo stems returned."""
    from models.separator import separate

    # 1-second mono at 44100 Hz
    audio = np.random.randn(44100).astype(np.float32)

    stems = separate(audio, sr=44100)

    assert stems.shape == (4, 2, 44100), f"Expected (4,2,44100), got {stems.shape}"
    assert stems.dtype == np.float32
    # Verify non-trivial output across all stems
    for i in range(4):
        assert not np.all(stems[i] == 0), f"Stem {i} is all zeros"


def test_separate_invalid_channels_raises_value_error():
    """Passing 3-channel or 5-channel audio raises ValueError."""
    from models.separator import separate

    audio_3ch = np.random.randn(3, 44100).astype(np.float32)
    with pytest.raises(ValueError, match="Expected 2 channels"):
        separate(audio_3ch, sr=44100)


def test_separate_invalid_sr_raises_value_error():
    """Sampling rate mismatch raises ValueError."""
    from models.separator import separate

    audio = np.random.randn(2, 44100).astype(np.float32)
    with pytest.raises(ValueError, match="Expected sr=44100"):
        separate(audio, sr=22050)


def test_separate_invalid_ndim_raises_value_error():
    """Invalid dimension (e.g. 3D array) raises ValueError."""
    from models.separator import separate

    audio_3d = np.random.randn(1, 2, 44100).astype(np.float32)
    with pytest.raises(ValueError, match="audio must be 1D"):
        separate(audio_3d, sr=44100)


