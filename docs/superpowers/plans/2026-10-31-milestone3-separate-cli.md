# Milestone 3: End-to-End Separation CLI & Pipeline Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/separate.py` (pure helper `separate_file()` + CLI) and `tests/test_separate_cli.py` (TDD tests with mocked HDemucs2Stem) for 2-stem vocal/accompaniment separation using the existing `HDemucs2Stem` module.

**Architecture:** Pure function `separate_file()` handles audio I/O, validation, chunking, and model inference. CLI wraps it with `argparse`. Tests use synthetic float32 sine waves and monkeypatch `HDemucs2Stem` to avoid heavy weight downloads. All dtypes explicitly float32.

**Tech Stack:** Python 3.13, soundfile, numpy, torch, demucs-infer (HDemucs2Stem from models/separator.py), argparse, pytest, pytest-mock.

**Spec:** This plan implements the Milestone 3 requirements from the user's task description and CLAUDE.md conventions.

## Global Constraints

- Python 3.13, Windows, venv at `./.venv/`
- All synthetic fixtures MUST use `np.float32` / `torch.float32` explicitly
- Audio I/O: `soundfile.read(..., dtype='float32', always_2d=True)` and `soundfile.write(..., samplerate=44100, subtype='PCM_16')`
- `HDemucs2Stem(freeze_encoders=False, checkpoint_path=checkpoint_path)` — `freeze_encoders` made explicit for inference (backbone loaded; no encoder freeze needed for separation); verified against `models/separator.py:77-81`
- Sample rate must be 44100 Hz (validate or resample)
- Model checkpoints gitignored (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`)
- Formatter: `black`
- Strict TDD: write failing test first, then implementation
- Mock/fake model in tests so they run <2 seconds without downloading weights
- Verify output files created, non-empty, shapes/channels/sr match input
- Graceful `FileNotFoundError` on invalid input paths

---

### Task 1: Create `tests/test_separate_cli.py` — TDD Tests First

**Files:**
- Create: `tests/test_separate_cli.py`

**Interfaces:**
- Consumes: None (first task)
- Produces: Test suite defining expected behavior of `scripts.separate.separate_file` and CLI

- [ ] **Step 1.1: Write the failing test for `separate_file` basic functionality**

```python
"""TDD tests for scripts/separate.py - End-to-End Separation CLI & Pipeline."""
import sys
from pathlib import Path
import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSeparateFile:
    """Tests for the pure helper function separate_file."""

    @pytest.fixture
    def synthetic_stereo_audio(self, tmp_path):
        """Create a 1-second stereo float32 sine wave at 44100 Hz."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Different frequencies for L and R channels
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # A4
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # A5
        audio = np.stack([left, right], axis=0)  # (2, N)
        path = tmp_path / "input.wav"
        import soundfile as sf
        sf.write(path, audio.T, sr, subtype="PCM_16")
        return path, audio, sr

    def test_separate_file_creates_outputs(self, synthetic_stereo_audio, tmp_path, monkeypatch):
        """separate_file writes vocals.wav and accompaniment.wav, returns dict of paths."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem

        input_path, audio, sr = synthetic_stereo_audio
        output_dir = tmp_path / "stems"

        # Mock HDemucs2Stem to avoid loading real weights
        def mock_forward(self, mix):
            # mix: (B, 2, T) -> return (B, 2, 2, T) [vocals, accompaniment]
            B, C, T = mix.shape
            # Return zeros for vocals, copy of input for accompaniment (sanity)
            out = torch.zeros((B, 2, C, T), dtype=torch.float32)
            out[:, 1, :, :] = mix  # accompaniment = input
            return out

        import torch
        monkeypatch.setattr(HDemucs2Stem, "forward", mock_forward)
        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        result = separate_file(input_path, output_dir, checkpoint_path=None, device="cpu")

        # Verify return dict
        assert isinstance(result, dict)
        assert set(result.keys()) == {"vocals", "accompaniment"}
        assert result["vocals"].exists()
        assert result["accompaniment"].exists()

    def test_separate_file_output_shapes_and_sr_match_input(self, synthetic_stereo_audio, tmp_path, monkeypatch):
        """Output files have same shape (2, N) and sample rate 44100 as input."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem
        import torch

        input_path, audio, sr = synthetic_stereo_audio
        output_dir = tmp_path / "stems"

        def mock_forward(self, mix):
            B, C, T = mix.shape
            out = torch.zeros((B, 2, C, T), dtype=torch.float32)
            out[:, 1, :, :] = mix
            return out

        monkeypatch.setattr(HDemucs2Stem, "forward", mock_forward)
        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        result = separate_file(input_path, output_dir, checkpoint_path=None, device="cpu")

        # Read back and verify
        import soundfile as sf
        for stem_name in ("vocals", "accompaniment"):
            data, out_sr = sf.read(result[stem_name], dtype="float32", always_2d=True)
            data = data.T  # (N, C) -> (C, N)
            assert data.shape == audio.shape, f"{stem_name}: shape {data.shape} != {audio.shape}"
            assert out_sr == sr, f"{stem_name}: sr {out_sr} != {sr}"
            assert data.dtype == np.float32, f"{stem_name}: dtype {data.dtype} != float32"

    def test_separate_file_mono_input_duplicated_to_stereo(self, tmp_path, monkeypatch):
        """Mono input (1 channel) is duplicated to stereo before processing."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem
        import torch

        sr = 44100
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # (N,)
        input_path = tmp_path / "mono_input.wav"
        import soundfile as sf
        sf.write(input_path, mono, sr, subtype="PCM_16")  # soundfile writes (N,) as mono

        output_dir = tmp_path / "stems"

        def mock_forward(self, mix):
            B, C, T = mix.shape
            # Verify C == 2 (stereo)
            assert C == 2, f"Expected stereo input to model, got {C} channels"
            out = torch.zeros((B, 2, C, T), dtype=torch.float32)
            out[:, 1, :, :] = mix
            return out

        monkeypatch.setattr(HDemucs2Stem, "forward", mock_forward)
        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        result = separate_file(input_path, output_dir, checkpoint_path=None, device="cpu")

        # Output should be stereo
        import soundfile as sf
        for stem_name in ("vocals", "accompaniment"):
            data, _ = sf.read(result[stem_name], dtype="float32", always_2d=True)
            assert data.shape[1] == 2, f"{stem_name}: expected 2 channels, got {data.shape[1]}"

    def test_separate_file_invalid_input_raises_filenotfound(self, tmp_path, monkeypatch):
        """FileNotFoundError raised for non-existent input path."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem
        import torch

        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            separate_file(tmp_path / "nonexistent.wav", tmp_path / "out")

    def test_separate_file_invalid_sr_raises_valueerror(self, tmp_path, monkeypatch):
        """ValueError raised if input sample rate != 44100."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem
        import torch

        # Create 22050 Hz audio
        sr = 22050
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = np.stack([np.sin(2*np.pi*440*t), np.sin(2*np.pi*880*t)], axis=0).astype(np.float32)
        input_path = tmp_path / "wrong_sr.wav"
        import soundfile as sf
        sf.write(input_path, audio.T, sr, subtype="PCM_16")

        output_dir = tmp_path / "stems"
        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        with pytest.raises(ValueError, match="Sample rate must be 44100 Hz"):
            separate_file(input_path, output_dir)

    def test_separate_file_chunked_processing(self, tmp_path, monkeypatch):
        """Long audio is split into chunks when chunk_duration_sec provided."""
        from scripts.separate import separate_file
        from models.separator import HDemucs2Stem
        import torch

        # 12 seconds at 44100 Hz
        sr = 44100
        duration = 12.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.stack([
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            np.sin(2 * np.pi * 880 * t).astype(np.float32),
        ], axis=0)
        input_path = tmp_path / "long_input.wav"
        import soundfile as sf
        sf.write(input_path, audio.T, sr, subtype="PCM_16")

        output_dir = tmp_path / "stems"
        call_count = {"n": 0}

        def mock_forward(self, mix):
            call_count["n"] += 1
            B, C, T = mix.shape
            out = torch.zeros((B, 2, C, T), dtype=torch.float32)
            out[:, 1, :, :] = mix
            return out

        monkeypatch.setattr(HDemucs2Stem, "forward", mock_forward)
        monkeypatch.setattr(HDemucs2Stem, "__init__", lambda self, **kwargs: None)

        # Chunk at 5 seconds -> should process 3 chunks (5s + 5s + 2s)
        result = separate_file(input_path, output_dir, checkpoint_path=None, device="cpu", chunk_duration_sec=5.0)

        # Verify 3 forward calls
        assert call_count["n"] == 3, f"Expected 3 chunk calls, got {call_count['n']}"

        # Output should be full 12 seconds
        import soundfile as sf
        data, out_sr = sf.read(result["vocals"], dtype="float32", always_2d=True)
        assert data.shape[0] == int(sr * duration), f"Output length mismatch: {data.shape[0]} != {int(sr * duration)}"
```

- [ ] **Step 1.2: Write the failing test for CLI argument parsing**

```python
class TestSeparateCLI:
    """Tests for the CLI argument parsing and main entry point."""

    def test_cli_parses_required_args(self, monkeypatch):
        """CLI accepts --input/-i, --output-dir/-o, --checkpoint/-c, --device."""
        from scripts.separate import main
        import sys

        # Mock separate_file to avoid actual processing
        def mock_separate_file(input_path, output_dir, *, checkpoint_path=None, device="cpu", chunk_duration_sec=None):
            return {"vocals": Path(output_dir) / "vocals.wav", "accompaniment": Path(output_dir) / "accompaniment.wav"}

        monkeypatch.setattr("scripts.separate.separate_file", mock_separate_file)

        # Simulate CLI args
        test_args = ["separate.py", "-i", "input.wav", "-o", "stems/", "--device", "cpu"]
        monkeypatch.setattr(sys, "argv", test_args)

        # Should not raise
        main()

    def test_cli_defaults_output_dir_to_stems(self, monkeypatch):
        """Default output directory is 'stems/' when -o not provided."""
        from scripts.separate import main
        import sys

        captured = {"output_dir": None}

        def mock_separate_file(input_path, output_dir, *, checkpoint_path=None, device="cpu", chunk_duration_sec=None):
            captured["output_dir"] = output_dir
            return {"vocals": Path(output_dir) / "vocals.wav", "accompaniment": Path(output_dir) / "accompaniment.wav"}

        monkeypatch.setattr("scripts.separate.separate_file", mock_separate_file)

        test_args = ["separate.py", "-i", "input.wav"]
        monkeypatch.setattr(sys, "argv", test_args)

        main()
        assert captured["output_dir"] == Path("stems")

    def test_cli_exits_with_error_on_missing_input(self, monkeypatch, capsys):
        """CLI exits with error code and message when --input not provided."""
        from scripts.separate import main
        import sys
        import pytest

        test_args = ["separate.py"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0

        captured = capsys.readouterr()
        assert "required" in captured.err.lower() or "error" in captured.err.lower()
```

- [ ] **Step 1.3: Run tests to verify they fail**

```bash
cd C:\Users\Samar\CS_F407_PROJECT\Computational-Musicology-for-Indian-Art-Music
.\.venv\Scripts\Activate.ps1
pytest tests/test_separate_cli.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'scripts.separate'` or similar.

- [ ] **Step 1.4: Commit test file**

```bash
git add tests/test_separate_cli.py
git commit -m "test: add TDD tests for scripts/separate.py (Milestone 3)"
```

---

### Task 2: Create `scripts/separate.py` — Implementation

**Files:**
- Create: `scripts/separate.py`

**Interfaces:**
- Consumes: `models.separator.HDemucs2Stem`, `soundfile`, `numpy`, `torch`
- Produces: `separate_file()` function, `main()` CLI entry point

- [ ] **Step 2.1: Write minimal `separate_file()` implementation**

```python
#!/usr/bin/env python
"""End-to-End Separation CLI & Pipeline — Milestone 3.

Pure helper `separate_file()` + CLI wrapper for 2-stem (vocals/accompaniment)
separation using HDemucs2Stem.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import torch

# Ensure repo root on path for models import
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.separator import HDemucs2Stem


TARGET_SR = 44100


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Resample audio to target sample rate if needed."""
    if sr == target_sr:
        return audio, sr
    # Use librosa for high-quality resampling
    import librosa
    # audio is (channels, samples)
    resampled = np.stack([
        librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr)
        for ch in range(audio.shape[0])
    ], axis=0)
    return resampled.astype(np.float32, copy=False), target_sr


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is stereo (2, N). If mono (N,) or (1, N), duplicate to stereo."""
    if audio.ndim == 1:
        # Mono 1D -> (2, N)
        return np.tile(audio[None, :], (2, 1))
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            # Mono 2D (1, N) -> (2, N)
            return np.tile(audio, (2, 1))
        elif audio.shape[0] == 2:
            return audio
        else:
            raise ValueError(f"Expected 1 or 2 channels, got {audio.shape[0]}")
    else:
        raise ValueError(f"Audio must be 1D or 2D, got ndim={audio.ndim}")


def _process_chunk(model: HDemucs2Stem, chunk: np.ndarray, device: torch.device) -> np.ndarray:
    """Process a single chunk through the model.

    Args:
        model: HDemucs2Stem in eval mode
        chunk: (2, chunk_samples) float32
        device: torch device

    Returns:
        (2, 2, chunk_samples) float32 — [vocals, accompaniment], each stereo
    """
    # Add batch dim: (1, 2, T)
    mix = torch.from_numpy(chunk.astype(np.float32, copy=False)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(mix)  # (1, 2, 2, T)
    return out[0].cpu().numpy()  # (2, 2, T)


def separate_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    chunk_duration_sec: float | None = None,
) -> dict[str, Path]:
    """Separate a single audio file into vocals and accompaniment stems.

    Parameters
    ----------
    input_path : str | Path
        Path to input .wav file (mono or stereo, any sample rate).
    output_dir : str | Path
        Directory to write output stems (created if not exists).
    checkpoint_path : str | Path | None, optional
        Path to fine-tuned .pt checkpoint. If None, uses pretrained HDemucs.
    device : str, default "cpu"
        "cpu" or "cuda".
    chunk_duration_sec : float | None, default None
        If provided, slice input into chunks of this duration (seconds) to
        avoid OOM on long audio. Chunks are processed sequentially and
        concatenated.

    Returns
    -------
    dict[str, Path]
        {"vocals": Path, "accompaniment": Path} pointing to written files.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If sample rate != 44100 and resampling not implemented, or invalid audio.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    # soundfile returns (N, C) -> transpose to (C, N)
    audio = audio.T

    # Resample if needed
    audio, sr = _resample_if_needed(audio, sr, TARGET_SR)

    # Ensure stereo
    audio = _ensure_stereo(audio)

    # Validate final sample rate
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate must be {TARGET_SR} Hz after resampling, got {sr}")

    # Load model
    torch_device = torch.device(device)
    model = HDemucs2Stem(checkpoint_path=checkpoint_path)
    model.to(torch_device)
    model.eval()

    # Process
    n_samples = audio.shape[1]

    if chunk_duration_sec is not None and chunk_duration_sec > 0:
        chunk_samples = int(chunk_duration_sec * TARGET_SR)
        # Process in chunks
        vocals_chunks = []
        accompaniment_chunks = []

        for start in range(0, n_samples, chunk_samples):
            end = min(start + chunk_samples, n_samples)
            chunk = audio[:, start:end]

            # Pad last chunk if shorter
            if chunk.shape[1] < chunk_samples:
                pad_width = chunk_samples - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode="constant")

            out = _process_chunk(model, chunk, torch_device)  # (2, 2, T)

            # Remove padding from last chunk
            actual_len = end - start
            vocals_chunks.append(out[0, :, :actual_len])
            accompaniment_chunks.append(out[1, :, :actual_len])

        vocals = np.concatenate(vocals_chunks, axis=1)
        accompaniment = np.concatenate(accompaniment_chunks, axis=1)
    else:
        # Process entire audio at once
        out = _process_chunk(model, audio, torch_device)  # (2, 2, T)
        vocals = out[0]  # (2, T)
        accompaniment = out[1]  # (2, T)

    # Write outputs (soundfile expects (N, C))
    vocals_path = output_dir / "vocals.wav"
    accompaniment_path = output_dir / "accompaniment.wav"

    sf.write(vocals_path, vocals.T, TARGET_SR, subtype="PCM_16")
    sf.write(accompaniment_path, accompaniment.T, TARGET_SR, subtype="PCM_16")

    return {"vocals": vocals_path, "accompaniment": accompaniment_path}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="2-stem vocal/accompaniment separation using HDemucs2Stem"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input .wav file (polyphonic mix)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="stems",
        help="Output directory for separated stems (default: stems/)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Optional path to fine-tuned .pt checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on (default: cpu)"
    )
    # Note: chunk_duration_sec not exposed in CLI for simplicity; can add later

    args = parser.parse_args()

    try:
        result = separate_file(
            input_path=args.input,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
            chunk_duration_sec=None,  # Not exposed in CLI yet
        )
        print(f"Separation complete:")
        print(f"  Vocals:        {result['vocals']}")
        print(f"  Accompaniment: {result['accompaniment']}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run tests to verify they pass**

```bash
cd C:\Users\Samar\CS_F407_PROJECT\Computational-Musicology-for-Indian-Art-Music
.\.venv\Scripts\Activate.ps1
pytest tests/test_separate_cli.py -v
```

Expected: All tests PASS.

- [ ] **Step 2.3: Run black formatter**

```bash
black scripts/separate.py tests/test_separate_cli.py
```

- [ ] **Step 2.4: Commit implementation**

```bash
git add scripts/separate.py
git commit -m "feat: add scripts/separate.py — End-to-End Separation CLI (Milestone 3)"
```

---

### Task 3: Integration Test & Final Verification

**Files:**
- Test: `tests/test_separate_cli.py` (already created)
- Modify: None

**Interfaces:**
- Consumes: Complete implementation from Task 2

- [ ] **Step 3.1: Run full test suite**

```bash
cd C:\Users\Samar\CS_F407_PROJECT\Computational-Musicology-for-Indian-Art-Music
.\.venv\Scripts\Activate.ps1
pytest tests/test_separate_cli.py -v
```

Expected: All tests PASS.

- [ ] **Step 3.2: Manual smoke test CLI**

```bash
cd C:\Users\Samar\CS_F407_PROJECT\Computational-Musicology-for-Indian-Art-Music
.\.venv\Scripts\Activate.ps1
python scripts/separate.py --help
```

Expected: Help text displays with all arguments.

- [ ] **Step 3.3: Verify no regression in existing tests**

```bash
pytest tests/ -v --ignore=tests/test_separate_cli.py
```

Expected: All existing tests still PASS.

- [ ] **Step 3.4: Final commit**

```bash
git add .
git commit -m "feat: Milestone 3 complete — End-to-End Separation CLI & tests"
```

---

## Execution Options

**Plan complete and saved to `docs/superpowers/plans/2026-10-31-milestone3-separate-cli.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**