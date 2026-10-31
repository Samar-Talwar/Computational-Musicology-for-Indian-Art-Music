"""Test suite for models/classifier.py"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add worktree root to path so `models` resolves
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTimbreCNN:
    """Tests for TimbreCNN architecture."""

    def test_forward_4d_input_shape(self):
        """Forward pass with 4D input (B, 1, 40, T) outputs (B, 10)."""
        from models.classifier import TimbreCNN

        model = TimbreCNN(num_classes=10)
        model.eval()

        batch = 4
        x = torch.randn(batch, 1, 40, 87, dtype=torch.float32)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch, 10), f"Expected ({batch}, 10), got {out.shape}"

    def test_forward_3d_input_shape(self):
        """Forward pass with 3D input (B, 40, T) outputs (B, 10)."""
        from models.classifier import TimbreCNN

        model = TimbreCNN(num_classes=10)
        model.eval()

        batch = 4
        x = torch.randn(batch, 40, 87, dtype=torch.float32)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch, 10), f"Expected ({batch}, 10), got {out.shape}"

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        from models.classifier import TimbreCNN

        model = TimbreCNN(num_classes=10)
        model.eval()

        x = torch.randn(2, 1, 40, 87, dtype=torch.float32)
        with torch.no_grad():
            out = model(x)

        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_gradient_flow(self):
        """Gradients flow through the model with a dummy loss."""
        from models.classifier import TimbreCNN

        model = TimbreCNN(num_classes=10)
        model.train()

        x = torch.randn(2, 1, 40, 87, dtype=torch.float32, requires_grad=True)
        target = torch.randint(0, 10, (2,))

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()

        # Check gradients exist on model parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

        # Check input gradient
        assert x.grad is not None, "No gradient for input"
        assert not torch.all(x.grad == 0), "Zero gradient for input"

    def test_num_classes_parameter(self):
        """num_classes parameter changes output dimension."""
        from models.classifier import TimbreCNN

        for num_classes in [5, 10, 20]:
            model = TimbreCNN(num_classes=num_classes)
            model.eval()
            x = torch.randn(1, 1, 40, 87, dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, num_classes), f"Expected (1, {num_classes}), got {out.shape}"


class TestExtractMFCC:
    """Tests for extract_mfcc feature extraction."""

    def test_output_shape(self):
        """Output shape is (40, 87)."""
        from models.classifier import extract_mfcc

        # 2 seconds of silence at 22050 Hz
        audio = np.zeros(44100, dtype=np.float32)
        mfcc = extract_mfcc(audio, sr=22050)

        assert mfcc.shape == (40, 87), f"Expected (40, 87), got {mfcc.shape}"

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        from models.classifier import extract_mfcc

        audio = np.zeros(44100, dtype=np.float32)
        mfcc = extract_mfcc(audio, sr=22050)

        assert mfcc.dtype == np.float32, f"Expected float32, got {mfcc.dtype}"

    def test_resample_from_44100(self):
        """Input at 44100 Hz is resampled to 22050 Hz."""
        from models.classifier import extract_mfcc

        # 2 seconds at 44100 Hz
        audio = np.zeros(88200, dtype=np.float32)
        mfcc = extract_mfcc(audio, sr=44100)

        assert mfcc.shape == (40, 87), f"Expected (40, 87), got {mfcc.shape}"

    def test_pad_short_audio(self):
        """Audio shorter than 2 seconds is zero-padded."""
        from models.classifier import extract_mfcc

        # 1 second at 22050 Hz
        audio = np.zeros(22050, dtype=np.float32)
        mfcc = extract_mfcc(audio, sr=22050)

        assert mfcc.shape == (40, 87), f"Expected (40, 87), got {mfcc.shape}"

    def test_trim_long_audio(self):
        """Audio longer than 2 seconds is trimmed."""
        from models.classifier import extract_mfcc

        # 3 seconds at 22050 Hz
        audio = np.zeros(66150, dtype=np.float32)
        mfcc = extract_mfcc(audio, sr=22050)

        assert mfcc.shape == (40, 87), f"Expected (40, 87), got {mfcc.shape}"

    def test_non_silent_audio(self):
        """Non-silent audio produces non-zero MFCCs."""
        from models.classifier import extract_mfcc

        # Sine wave at 440 Hz
        t = np.linspace(0, 2, 44100, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        mfcc = extract_mfcc(audio, sr=22050)

        assert mfcc.shape == (40, 87)
        assert mfcc.dtype == np.float32
        assert not np.all(mfcc == 0), "MFCC should not be all zeros for tonal audio"


class TestINSTRUMENT_CLASSES:
    """Tests for INSTRUMENT_CLASSES constant."""

    def test_length_and_content(self):
        """INSTRUMENT_CLASSES has 10 expected classes."""
        from models.classifier import INSTRUMENT_CLASSES

        assert len(INSTRUMENT_CLASSES) == 10
        expected = [
            "bass",
            "drums",
            "flute",
            "guitar",
            "piano",
            "sarod",
            "sitar",
            "tabla",
            "veena",
            "violin",
        ]
        assert INSTRUMENT_CLASSES == expected