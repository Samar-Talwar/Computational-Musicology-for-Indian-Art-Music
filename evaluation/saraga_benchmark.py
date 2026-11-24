"""Saraga multitrack in-the-wild noise-resilience and bleed-audit benchmark.

Evaluates stem bleed (percussive bleed into vocals, string confidence in accompaniment)
and HPSS harmonic-to-percussive ratio on unlabelled concert recordings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from models.classifier import INSTRUMENT_CLASSES, TimbreCNN, extract_mfcc
from models.separator import HDemucs2Stem
from preprocessing.hpss import hpss

TARGET_SR: int = 44100
DEFAULT_CHUNK_DURATION_SEC: float = 6.0
DEFAULT_MAX_CHUNKS_PER_TRACK: int = 15
SILENCE_RMS_THRESHOLD: float = 0.02


def _serialize(obj: Any) -> Any:
    """Recursively convert NumPy and Path types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def load_saraga_excerpts(
    saraga_dir: Path | str,
    segment_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    max_chunks_per_track: int = DEFAULT_MAX_CHUNKS_PER_TRACK,
) -> list[dict[str, Any]]:
    """Scan saraga_dir for .wav audio, validate 44.1kHz stereo, and slice active non-silent chunks.

    Parameters
    ----------
    saraga_dir : Path | str
        Root directory containing Saraga audio files.
    segment_duration_sec : float, default 6.0
        Duration in seconds of each audio chunk.
    max_chunks_per_track : int, default 15
        Maximum number of active chunks extracted per track.

    Returns
    -------
    list[dict[str, Any]]
        List of chunk dictionaries with track metadata, audio tensor (2, N) float32, and RMS.

    Raises
    ------
    FileNotFoundError
        If saraga_dir does not exist.
    ValueError
        If any discovered file is not 44,100 Hz or cannot be normalized to stereo.
    """
    saraga_path = Path(saraga_dir)
    if not saraga_path.exists():
        raise FileNotFoundError(f"Saraga directory not found: {saraga_path}")

    wav_files = sorted(
        [p for p in saraga_path.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    )

    all_chunks: list[dict[str, Any]] = []
    chunk_samples = int(segment_duration_sec * TARGET_SR)

    for wav_path in wav_files:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        if sr != TARGET_SR:
            raise ValueError(
                f"Sample rate mismatch: expected {TARGET_SR} Hz, got {sr} Hz in {wav_path}"
            )

        # Transpose to (channels, samples)
        audio = audio.T
        if audio.shape[0] == 1:
            audio = np.tile(audio, (2, 1)).astype(np.float32, copy=False)
        elif audio.shape[0] == 2:
            audio = audio.astype(np.float32, copy=False)
        else:
            raise ValueError(
                f"Expected 1 or 2 channels, got {audio.shape[0]} channels in {wav_path}"
            )

        total_samples = audio.shape[1]
        track_chunk_count = 0

        for start in range(0, total_samples - chunk_samples + 1, chunk_samples):
            if track_chunk_count >= max_chunks_per_track:
                break
            end = start + chunk_samples
            chunk = audio[:, start:end]

            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms > SILENCE_RMS_THRESHOLD:
                all_chunks.append(
                    {
                        "track_id": wav_path.stem,
                        "track_path": str(wav_path),
                        "chunk_idx": track_chunk_count,
                        "start_sec": float(start / TARGET_SR),
                        "end_sec": float(end / TARGET_SR),
                        "audio": chunk.copy(),
                        "rms": rms,
                    }
                )
                track_chunk_count += 1

    return all_chunks


def audit_saraga_noise_resilience(
    chunks: list[dict[str, Any]],
    sep_checkpoint: Path | str | None = None,
    clf_checkpoint: Path | str | None = None,
    device: str = "cpu",
    model: nn.Module | None = None,
    classifier: nn.Module | None = None,
) -> dict[str, Any]:
    """Run 2-stem separation, timbre bleed classification, and HPSS ratio audit on audio chunks.

    Parameters
    ----------
    chunks : list[dict[str, Any]]
        Extracted active chunks from load_saraga_excerpts.
    sep_checkpoint : Path | str | None, optional
        Path to fine-tuned separator checkpoint.
    clf_checkpoint : Path | str | None, optional
        Path to trained classifier checkpoint.
    device : str, default "cpu"
        PyTorch execution device.
    model : nn.Module | None, optional
        Injected separator module for offline tests.
    classifier : nn.Module | None, optional
        Injected classifier module for offline tests.

    Returns
    -------
    dict[str, Any]
        Dictionary of per-chunk and aggregated bleed/resilience metrics.
    """
    torch_dev = torch.device(device)

    # 1. Initialize separator model
    if model is None:
        model = HDemucs2Stem(freeze_encoders=True, checkpoint_path=sep_checkpoint)
    model.to(torch_dev).eval()

    # 2. Initialize classifier model
    if classifier is None:
        classifier = TimbreCNN(num_classes=len(INSTRUMENT_CLASSES))
        if clf_checkpoint is not None:
            ckpt = torch.load(clf_checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            classifier.load_state_dict(state)
    classifier.to(torch_dev).eval()

    tabla_idx = INSTRUMENT_CLASSES.index("tabla")
    drums_idx = INSTRUMENT_CLASSES.index("drums")
    violin_idx = INSTRUMENT_CLASSES.index("violin")
    veena_idx = INSTRUMENT_CLASSES.index("veena")

    per_chunk_results: list[dict[str, Any]] = []

    for chunk_info in chunks:
        chunk_audio = chunk_info["audio"]  # shape (2, chunk_samples)
        mix_tensor = torch.from_numpy(chunk_audio).unsqueeze(0).to(torch_dev)  # (1, 2, T)

        with torch.no_grad():
            out = model(mix_tensor)  # (1, 2, 2, T) -> [vocals, accompaniment]

        vocals = out[0, 0].cpu().numpy().astype(np.float32)  # (2, T)
        accompaniment = out[0, 1].cpu().numpy().astype(np.float32)  # (2, T)

        # Collapse to mono for feature extraction & DSP
        vocals_mono = np.mean(vocals, axis=0).astype(np.float32)
        acc_mono = np.mean(accompaniment, axis=0).astype(np.float32)

        # Multi-window MFCC extraction: split 6.0s into three 2.0s segments (88,200 samples @ 44.1kHz)
        def _classify_stem(mono_audio: np.ndarray) -> np.ndarray:
            sub_window_len = int(2.0 * TARGET_SR)  # 88,200 samples
            n_sub_windows = max(1, len(mono_audio) // sub_window_len)
            sub_probs: list[np.ndarray] = []

            for w_idx in range(n_sub_windows):
                w_start = w_idx * sub_window_len
                w_end = w_start + sub_window_len
                sub_audio = mono_audio[w_start:w_end]
                # extract_mfcc internally resamples 44100 -> 22050 and trims/pads to 44100
                mfcc = extract_mfcc(sub_audio, sr=TARGET_SR)  # (40, 87)
                x = torch.from_numpy(mfcc).unsqueeze(0).to(torch_dev)  # (1, 40, 87)
                with torch.no_grad():
                    logits = classifier(x)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                sub_probs.append(probs)

            return np.mean(sub_probs, axis=0)

        vocal_probs = _classify_stem(vocals_mono)
        acc_probs = _classify_stem(acc_mono)

        # Percussive bleed in vocals: tabla + drums
        vocal_percussive_bleed = float(vocal_probs[tabla_idx] + vocal_probs[drums_idx])
        # Indian string confidence in accompaniment: violin + veena
        acc_string_confidence = float(acc_probs[violin_idx] + acc_probs[veena_idx])

        # HPSS harmonic-to-percussive ratio
        def _compute_hpr(mono_audio: np.ndarray) -> tuple[float, float, float, float]:
            s_h, s_p = hpss(mono_audio, sr=TARGET_SR)
            e_h = float(np.sum(s_h**2))
            e_p = float(np.sum(s_p**2))
            ratio = float(e_h / (e_p + 1e-10))
            ratio_db = float(10.0 * np.log10((e_h + 1e-10) / (e_p + 1e-10)))
            return e_h, e_p, ratio, ratio_db

        v_eh, v_ep, v_hpr, v_hpr_db = _compute_hpr(vocals_mono)
        a_eh, a_ep, a_hpr, a_hpr_db = _compute_hpr(acc_mono)

        chunk_record = {
            "track_id": chunk_info["track_id"],
            "track_path": chunk_info["track_path"],
            "chunk_idx": chunk_info["chunk_idx"],
            "start_sec": float(chunk_info["start_sec"]),
            "end_sec": float(chunk_info["end_sec"]),
            "rms": float(chunk_info["rms"]),
            "vocal_percussive_bleed": vocal_percussive_bleed,
            "accompaniment_string_confidence": acc_string_confidence,
            "vocal_hpr_db": v_hpr_db,
            "accompaniment_hpr_db": a_hpr_db,
            "vocal_harmonic_energy": v_eh,
            "vocal_percussive_energy": v_ep,
            "accompaniment_harmonic_energy": a_eh,
            "accompaniment_percussive_energy": a_ep,
            "vocal_class_probabilities": {
                name: float(vocal_probs[idx]) for idx, name in enumerate(INSTRUMENT_CLASSES)
            },
            "accompaniment_class_probabilities": {
                name: float(acc_probs[idx]) for idx, name in enumerate(INSTRUMENT_CLASSES)
            },
        }
        per_chunk_results.append(chunk_record)

    total_chunks = len(per_chunk_results)
    if total_chunks == 0:
        return _serialize(
            {
                "total_chunks_audited": 0,
                "mean_vocal_percussive_bleed": 0.0,
                "mean_accompaniment_string_confidence": 0.0,
                "mean_vocal_hpr_db": 0.0,
                "mean_accompaniment_hpr_db": 0.0,
                "mean_vocal_class_probabilities": {name: 0.0 for name in INSTRUMENT_CLASSES},
                "mean_accompaniment_class_probabilities": {
                    name: 0.0 for name in INSTRUMENT_CLASSES
                },
                "per_chunk_results": [],
            }
        )

    mean_vocal_bleed = float(
        np.mean([r["vocal_percussive_bleed"] for r in per_chunk_results])
    )
    mean_acc_strings = float(
        np.mean([r["accompaniment_string_confidence"] for r in per_chunk_results])
    )
    mean_vocal_hpr = float(np.mean([r["vocal_hpr_db"] for r in per_chunk_results]))
    mean_acc_hpr = float(np.mean([r["accompaniment_hpr_db"] for r in per_chunk_results]))

    mean_vocal_probs = {
        name: float(
            np.mean([r["vocal_class_probabilities"][name] for r in per_chunk_results])
        )
        for name in INSTRUMENT_CLASSES
    }
    mean_acc_probs = {
        name: float(
            np.mean([r["accompaniment_class_probabilities"][name] for r in per_chunk_results])
        )
        for name in INSTRUMENT_CLASSES
    }

    result = {
        "total_chunks_audited": total_chunks,
        "mean_vocal_percussive_bleed": mean_vocal_bleed,
        "mean_accompaniment_string_confidence": mean_acc_strings,
        "mean_vocal_hpr_db": mean_vocal_hpr,
        "mean_accompaniment_hpr_db": mean_acc_hpr,
        "mean_vocal_class_probabilities": mean_vocal_probs,
        "mean_accompaniment_class_probabilities": mean_acc_probs,
        "per_chunk_results": per_chunk_results,
    }

    return _serialize(result)
