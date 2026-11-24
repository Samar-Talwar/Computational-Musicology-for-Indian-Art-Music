"""Unit tests for evaluation/saraga_benchmark.py and scripts/run_saraga_eval.py.

Strict offline TDD tests using tmp_path synthetic 44.1 kHz float32 WAVs and lightweight dummy models
to guarantee test execution in <2 seconds without requiring real Saraga audio or weights on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.saraga_benchmark import (
    audit_saraga_noise_resilience,
    load_saraga_excerpts,
)
from models.classifier import INSTRUMENT_CLASSES
from scripts.run_saraga_eval import run_saraga_eval


class DummySeparator(nn.Module):
    """Fast dummy 2-stem separator returning predictable (B, 2, 2, T) tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        # mix: (B, 2, T)
        b, c, t = mix.shape
        # Create vocal and accompaniment stems by scaling mix
        vocals = mix * 0.8
        accompaniment = mix * 0.5
        # Return (B, 2, 2, T) where dim 1 is [vocals, accompaniment]
        return torch.stack([vocals, accompaniment], dim=1)


class DummyClassifier(nn.Module):
    """Fast dummy instrument classifier returning predictable (B, 10) logits."""

    def __init__(self) -> None:
        super().__init__()
        self.num_classes = len(INSTRUMENT_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, T) or (B, 40, T)
        b = x.shape[0]
        logits = torch.zeros(b, self.num_classes, dtype=torch.float32)
        # Put high energy on tabla (idx 7) and drums (idx 1), and violin (idx 9)
        logits[:, 7] = 2.0  # tabla
        logits[:, 1] = 1.0  # drums
        logits[:, 9] = 3.0  # violin
        logits[:, 8] = 1.5  # veena
        return logits


def _create_synthetic_wav(
    file_path: Path,
    sr: int = 44100,
    duration_sec: float = 12.0,
    channels: int = 2,
    amplitude: float = 0.1,
) -> Path:
    """Generate synthetic float32 sine wave WAV file."""
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
    # 440 Hz tone
    mono = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    if channels == 1:
        audio = mono
    elif channels == 2:
        # Stereo with slightly different channel amplitudes
        audio = np.stack([mono, mono * 0.9], axis=1)
    else:
        audio = np.stack([mono] * channels, axis=1)

    sf.write(file_path, audio, sr, subtype="FLOAT")
    return file_path


class TestSaragaBenchmark:
    """Test suite verifying Saraga excerpt loading and noise-resilience audit."""

    def test_load_saraga_excerpts_active_chunks(self, tmp_path: Path) -> None:
        """Verify active 6.0s chunks are extracted with (2, 264600) float32 shape."""
        # 12-second audio -> exactly 2 non-overlapping 6.0s chunks
        wav_path = tmp_path / "track1.wav"
        _create_synthetic_wav(wav_path, sr=44100, duration_sec=12.0, amplitude=0.1)

        chunks = load_saraga_excerpts(tmp_path, segment_duration_sec=6.0)
        assert len(chunks) == 2

        for i, chunk in enumerate(chunks):
            assert chunk["track_id"] == "track1"
            assert chunk["chunk_idx"] == i
            assert chunk["audio"].shape == (2, 264600)
            assert chunk["audio"].dtype == np.float32
            assert chunk["rms"] > 0.02
            assert chunk["start_sec"] == pytest.approx(i * 6.0, abs=1e-3)
            assert chunk["end_sec"] == pytest.approx((i + 1) * 6.0, abs=1e-3)

    def test_load_saraga_excerpts_invalid_sr(self, tmp_path: Path) -> None:
        """Verify non-44100 Hz audio raises ValueError."""
        wav_path = tmp_path / "invalid_sr.wav"
        _create_synthetic_wav(wav_path, sr=22050, duration_sec=6.0, amplitude=0.1)

        with pytest.raises(ValueError, match="Sample rate mismatch"):
            load_saraga_excerpts(tmp_path)

    def test_load_saraga_excerpts_silent_skip(self, tmp_path: Path) -> None:
        """Verify silent chunks (RMS <= 0.02) are skipped."""
        wav_path = tmp_path / "silent.wav"
        # Zero amplitude -> RMS = 0.0 <= 0.02
        _create_synthetic_wav(wav_path, sr=44100, duration_sec=12.0, amplitude=0.0)

        chunks = load_saraga_excerpts(tmp_path, segment_duration_sec=6.0)
        assert len(chunks) == 0

    def test_load_saraga_excerpts_max_chunks(self, tmp_path: Path) -> None:
        """Verify max_chunks_per_track caps extracted chunks."""
        wav_path = tmp_path / "long_track.wav"
        # 30-second audio has 5 potential 6.0s chunks
        _create_synthetic_wav(wav_path, sr=44100, duration_sec=30.0, amplitude=0.1)

        chunks = load_saraga_excerpts(tmp_path, segment_duration_sec=6.0, max_chunks_per_track=2)
        assert len(chunks) == 2

    def test_load_saraga_excerpts_mono_expanded_to_stereo(self, tmp_path: Path) -> None:
        """Verify 1-channel mono is duplicated to (2, N) stereo float32."""
        wav_path = tmp_path / "mono.wav"
        _create_synthetic_wav(wav_path, sr=44100, duration_sec=6.0, channels=1, amplitude=0.1)

        chunks = load_saraga_excerpts(tmp_path, segment_duration_sec=6.0)
        assert len(chunks) == 1
        assert chunks[0]["audio"].shape == (2, 264600)
        assert chunks[0]["audio"].dtype == np.float32
        # Check channel duplication
        assert np.array_equal(chunks[0]["audio"][0], chunks[0]["audio"][1])

    def test_load_saraga_excerpts_missing_dir(self, tmp_path: Path) -> None:
        """Verify missing directory raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_saraga_excerpts(non_existent)

    def test_audit_saraga_noise_resilience_dummy_models(self, tmp_path: Path) -> None:
        """Verify complete audit pipeline with dummy models executes offline in <1s with valid schema."""
        wav_path = tmp_path / "song.wav"
        _create_synthetic_wav(wav_path, sr=44100, duration_sec=6.0, amplitude=0.1)
        chunks = load_saraga_excerpts(tmp_path, segment_duration_sec=6.0)
        assert len(chunks) == 1

        dummy_sep = DummySeparator()
        dummy_clf = DummyClassifier()

        result = audit_saraga_noise_resilience(
            chunks,
            model=dummy_sep,
            classifier=dummy_clf,
            device="cpu",
        )

        # 1. Output must be a dict
        assert isinstance(result, dict)
        assert result["total_chunks_audited"] == 1
        assert "mean_vocal_percussive_bleed" in result
        assert "mean_accompaniment_string_confidence" in result
        assert "mean_vocal_hpr_db" in result
        assert "mean_accompaniment_hpr_db" in result
        assert "mean_vocal_class_probabilities" in result
        assert "mean_accompaniment_class_probabilities" in result
        assert len(result["per_chunk_results"]) == 1

        # 2. Values must be non-negative probabilities / valid floats
        assert 0.0 <= result["mean_vocal_percussive_bleed"] <= 1.0
        assert 0.0 <= result["mean_accompaniment_string_confidence"] <= 1.0
        assert isinstance(result["mean_vocal_hpr_db"], float)
        assert isinstance(result["mean_accompaniment_hpr_db"], float)

        # 3. Clean JSON serialization test (guarantees native Python primitives, no numpy scalar errors)
        json_str = json.dumps(result)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["total_chunks_audited"] == 1

    def test_audit_saraga_noise_resilience_empty_chunks(self) -> None:
        """Verify empty chunk input returns valid zero-count structure."""
        result = audit_saraga_noise_resilience(
            [],
            model=DummySeparator(),
            classifier=DummyClassifier(),
            device="cpu",
        )
        assert result["total_chunks_audited"] == 0
        assert result["mean_vocal_percussive_bleed"] == 0.0
        assert result["per_chunk_results"] == []
        # JSON serializable
        assert json.loads(json.dumps(result))["total_chunks_audited"] == 0

    def test_run_saraga_eval_cli(self, tmp_path: Path) -> None:
        """Verify run_saraga_eval end-to-end with output JSON creation."""
        saraga_dir = tmp_path / "saraga_raw"
        saraga_dir.mkdir()
        _create_synthetic_wav(saraga_dir / "concert1.wav", sr=44100, duration_sec=6.0, amplitude=0.1)

        out_json = tmp_path / "experiments" / "audit.json"

        # Mock audit_saraga_noise_resilience call or inject dummy model by patching
        from unittest.mock import patch

        dummy_sep = DummySeparator()
        dummy_clf = DummyClassifier()

        with patch(
            "evaluation.saraga_benchmark.HDemucs2Stem",
            return_value=dummy_sep,
        ), patch(
            "evaluation.saraga_benchmark.TimbreCNN",
            return_value=dummy_clf,
        ):
            res = run_saraga_eval(
                saraga_dir=saraga_dir,
                output_json=out_json,
                max_chunks=5,
                device="cpu",
            )

        assert out_json.exists()
        assert res["total_chunks_audited"] == 1
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_chunks_audited"] == 1
