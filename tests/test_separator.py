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

