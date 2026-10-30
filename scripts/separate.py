#!/usr/bin/env python
"""End-to-End Separation CLI & Pipeline — Milestone 3.
Pure helper `separate_file()` + CLI. 2-stem (vocals/accompaniment) via HDemucs2Stem.
Synthetic fixtures must be float32 (CLAUDE.md convention).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.separator import HDemucs2Stem

TARGET_SR = 44100


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Resample audio to target sample rate via librosa (per-channel loop — documented heuristic)."""
    if sr == target_sr:
        return audio.astype(np.float32, copy=False), sr
    import librosa  # declared dependency; lazy import per heuristic
    resampled = np.stack([
        librosa.resample(audio[ch], orig_sr=sr, target_sr=target_sr).astype(np.float32, copy=False)
        for ch in range(audio.shape[0])
    ], axis=0)
    return resampled, target_sr


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.tile(audio[None, :], (2, 1)).astype(np.float32, copy=False)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.tile(audio, (2, 1)).astype(np.float32, copy=False)
        if audio.shape[0] == 2:
            return audio.astype(np.float32, copy=False)
        raise ValueError(f"Expected 1 or 2 channels, got {audio.shape[0]}")
    raise ValueError(f"Audio must be 1D or 2D, got ndim={audio.ndim}")


def separate_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    chunk_duration_sec: float | None = None,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
    # Explicit SR validation before any resampling
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate must be {TARGET_SR} Hz, got {sr}")
    audio = audio.T  # (C, N)
    audio, sr = _resample_if_needed(audio, sr, TARGET_SR)
    audio = _ensure_stereo(audio)
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate must be {TARGET_SR} Hz, got {sr}")

    torch_dev = torch.device(device)
    # freeze_encoders=False explicit for inference (backbone loaded; no freeze needed)
    model = HDemucs2Stem(freeze_encoders=False, checkpoint_path=checkpoint_path)
    model.to(torch_dev).eval()

    n_samples = audio.shape[1]
    if chunk_duration_sec is not None and chunk_duration_sec > 0:
        chunk_samples = int(chunk_duration_sec * TARGET_SR)
        vocals_chunks, acc_chunks = [], []
        for start in range(0, n_samples, chunk_samples):
            end = min(start + chunk_samples, n_samples)
            chunk = audio[:, start:end]
            if chunk.shape[1] < chunk_samples:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_samples - chunk.shape[1])), mode="constant")
            mix = torch.from_numpy(chunk).unsqueeze(0).to(torch_dev)
            with torch.no_grad():
                out = model(mix)  # (B, 2, 2, T)
            actual_len = end - start
            vocals_chunks.append(out[0, 0, :, :actual_len].cpu().numpy())
            acc_chunks.append(out[0, 1, :, :actual_len].cpu().numpy())
        # Non-overlapping chunking — boundary artifacts possible (documented)
        vocals = np.concatenate(vocals_chunks, axis=1)
        accompaniment = np.concatenate(acc_chunks, axis=1)
    else:
        mix = torch.from_numpy(audio).unsqueeze(0).to(torch_dev)
        with torch.no_grad():
            out = model(mix)  # (B, 2, 2, T)
        vocals = out[0, 0, :, :].cpu().numpy()       # (2, T) — non-zero by model
        accompaniment = out[0, 1, :, :].cpu().numpy()  # (2, T)

    vocals_path = output_dir / "vocals.wav"
    accompaniment_path = output_dir / "accompaniment.wav"
    sf.write(vocals_path, vocals.T, TARGET_SR, subtype="PCM_16")
    sf.write(accompaniment_path, accompaniment.T, TARGET_SR, subtype="PCM_16")
    return {"vocals": vocals_path, "accompaniment": accompaniment_path}


def main():
    parser = argparse.ArgumentParser(description="2-stem separation using HDemucs2Stem")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input .wav mix (mono/stereo)")
    parser.add_argument("--output-dir", "-o", type=str, default="stems",
                        help="Output directory (default: stems/)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Fine-tuned .pt checkpoint")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--chunk-duration", type=float, default=None,
                        help="Chunk duration (sec) for long audio; non-overlapping — boundary artifacts possible")
    args = parser.parse_args()
    try:
        result = separate_file(
            args.input, args.output_dir,
            checkpoint_path=args.checkpoint, device=args.device,
            chunk_duration_sec=args.chunk_duration,
        )
        print("Separation complete:")
        print(f"  Vocals:        {result['vocals']}")
        print(f"  Accompaniment: {result['accompaniment']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
