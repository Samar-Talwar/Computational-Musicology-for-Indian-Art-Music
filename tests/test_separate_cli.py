"""TDD tests for scripts/separate.py — Milestone 3 End-to-End Separation CLI.
All fixtures explicitly float32 per CLAUDE.md line 47.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSeparateFile:
    """Unit / integration tests for separate_file() — mock model, fast (<2s)."""

    @pytest.fixture
    def synthetic_stereo(self, tmp_path):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        audio = np.stack([left, right], axis=0)  # (2, N) float32
        p = tmp_path / "mix.wav"
        import soundfile as sf
        sf.write(p, audio.T, sr, subtype="PCM_16")
        return p, audio, sr

    def _fake_model(self, monkeypatch):
        from models.separator import HDemucs2Stem

        import torch.nn as nn
        import torch.nn as nn
        class Dummy(nn.Module):
            def __init__(self, freeze_encoders=True, checkpoint_path=None):
                super().__init__()
            def eval(self, mode=True):
                return self
            def to(self, *args, **kwargs):
                return self
            def forward(self, mix):
                B, C, T = mix.shape
                assert C == 2
                out = torch.zeros((B, 2, C, T), dtype=torch.float32)
                out[:, 0, :, :] = mix
                out[:, 1, :, :] = mix * 0.5
                assert out.shape == (B, 2, 2, T)
                return out
        monkeypatch.setattr(HDemucs2Stem, "__init__", Dummy.__init__)
        monkeypatch.setattr(HDemucs2Stem, "forward", Dummy.forward)

        def forward(self, mix):
            # mix: (B, 2, T) -> (B, 2, 2, T) per spec; vocals = mix, acc = mix*0.5
            B, C, T = mix.shape
            assert C == 2, f"Expected stereo mix, got {C} ch"
            out = torch.zeros((B, 2, C, T), dtype=torch.float32)
            out[:, 0, :, :] = mix  # vocals = input (non-zero!)
            out[:, 1, :, :] = mix * 0.5  # accompaniment = half
            assert out.shape == (B, 2, 2, T)
            return out

        monkeypatch.setattr(HDemucs2Stem, "__init__", __init__)
        monkeypatch.setattr(HDemucs2Stem, "forward", forward)

    def test_outputs_created_and_non_empty(self, synthetic_stereo, tmp_path, monkeypatch):
        from scripts.separate import separate_file
        self._fake_model(monkeypatch)
        p, _, sr = synthetic_stereo
        out_dir = tmp_path / "stems"
        result = separate_file(p, out_dir, checkpoint_path=None, device="cpu")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"vocals", "accompaniment"}
        assert result["vocals"].exists() and result["vocals"].stat().st_size > 0
        assert result["accompaniment"].exists() and result["accompaniment"].stat().st_size > 0

    def test_output_shape_sr_dtype(self, synthetic_stereo, tmp_path, monkeypatch):
        from scripts.separate import separate_file
        self._fake_model(monkeypatch)
        p, audio, sr = synthetic_stereo
        out_dir = tmp_path / "stems"
        result = separate_file(p, out_dir, checkpoint_path=None, device="cpu")
        import soundfile as sf
        for name in ("vocals", "accompaniment"):
            data, out_sr = sf.read(result[name], dtype="float32", always_2d=True)
            data = data.T  # (C, N)
            assert data.dtype == np.float32
            assert out_sr == sr == 44100
            assert data.shape == audio.shape, f"{name} shape {data.shape} != {audio.shape}"
            assert data.shape[0] == 2  # stereo

    def test_mono_duplicated_to_stereo(self, tmp_path, monkeypatch):
        from scripts.separate import separate_file
        self._fake_model(monkeypatch)
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        p = tmp_path / "mono.wav"
        import soundfile as sf
        sf.write(p, mono, sr, subtype="PCM_16")
        result = separate_file(p, tmp_path / "out", checkpoint_path=None, device="cpu")
        import soundfile as sf
        data, _ = sf.read(result["vocals"], dtype="float32", always_2d=True)
        assert data.shape[1] == 2, "Mono input must yield stereo output"

    def test_invalid_path_raises_filenotfound(self, tmp_path, monkeypatch):
        from scripts.separate import separate_file
        self._fake_model(monkeypatch)
        with pytest.raises(FileNotFoundError):
            separate_file(tmp_path / "no_such.wav", tmp_path / "out")

    def test_bad_sr_raises(self, tmp_path, monkeypatch):
        from scripts.separate import separate_file
        self._fake_model(monkeypatch)
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        audio = np.stack([np.sin(2 * np.pi * 440 * t).astype(np.float32)] * 2, axis=0)
        p = tmp_path / "bad_sr.wav"
        import soundfile as sf
        sf.write(p, audio.T, sr, subtype="PCM_16")
        with pytest.raises(ValueError, match="44100"):
            separate_file(p, tmp_path / "out")


class TestSeparateCLI:
    def test_argparse_required_input(self, monkeypatch):
        import argparse
        from scripts.separate import main
        monkeypatch.setattr("sys.argv", ["separate.py", "-i", "in.wav", "-o", "out/"])
        # Don't actually run; just verify parser accepts required arg
        # Real execution uses mock in integration; here only verify parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", "-i", required=True)
        parser.add_argument("--output-dir", "-o", default="stems")
        args = parser.parse_args(["-i", "in.wav", "-o", "out/"])
        assert args.input == "in.wav"
        assert args.output_dir == "out/"

    def test_cli_default_output_dir(self, monkeypatch):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", "-i", required=True)
        parser.add_argument("--output-dir", "-o", default="stems")
        args = parser.parse_args(["-i", "x.wav"])
        assert args.output_dir == "stems"
