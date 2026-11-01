#!/usr/bin/env python
"""Staging dataset for Colab fine-tuning — Milestone 2 Step 4."""
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import csv
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import soundfile as sf

from utils.sanidha import index_sanidha, split_sanidha, load_song, load_logical_stem, SongRecord
from evaluation.metrics import collapse_to_two_stem


def _energy(x: np.ndarray) -> float:
    return float(np.dot(x.ravel(), x.ravel()))


def stage_song(
    record: SongRecord,
    output_dir: Path,
    *,
    chunk_duration_sec: float = 6.0,
    include_tanpura: bool = True,
) -> list[dict[str, Any]]:
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be > 0")
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Determine total duration from first stem header
    first_path = next(iter(record.stem_files.values()))[0]
    info = sf.info(first_path)
    total_duration = info.duration
    sr = info.samplerate

    chunks_meta: list[dict[str, Any]] = []
    skipped_silent = 0
    window_idx = 0

    start_sec = 0.0
    while start_sec + chunk_duration_sec <= total_duration + 1e-6:
        # Load 4 primary stems
        vocals = load_logical_stem(record, "vocals", start_sec=start_sec, duration_sec=chunk_duration_sec, target_sr=sr)
        violin = load_logical_stem(record, "violin", start_sec=start_sec, duration_sec=chunk_duration_sec, target_sr=sr)
        mridangam = load_logical_stem(record, "mridangam", start_sec=start_sec, duration_sec=chunk_duration_sec, target_sr=sr)

        has_ghatam_file = "ghatam" in record.stem_files
        if has_ghatam_file:
            ghatam = load_logical_stem(record, "ghatam", start_sec=start_sec, duration_sec=chunk_duration_sec, target_sr=sr)
        else:
            # Silent array matching first stem shape (2, N) float32 - ensure C-contiguous
            n_samples = vocals.shape[1]
            ghatam = np.zeros((2, n_samples), dtype=np.float32, order="C")

        # Energy filter across all 4 instruments (RMS <= 1e-12)
        # Only skip if ALL primary instruments are silent; a silent ghatam is allowed.
        primary = (vocals, violin, mridangam)
        silent = False
        for stem in primary:
            energy = float(np.dot(stem.ravel(), stem.ravel()))
            if energy <= 1e-12:
                silent = True
                break
        if silent:
            skipped_silent += 1
            start_sec += chunk_duration_sec
            window_idx += 1
            continue

        chunk_id = f"C{record.concert_id:02d}_{record.song_id}_chunk{window_idx:04d}"
        chunk_dir = chunks_dir / chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Mix = sum of 4 stems (channels-first)
        mix = (vocals + violin + mridangam + ghatam).astype(np.float32, copy=False)

        # Write 5 audio files (channels-first -> soundfile expects (N, C))
        # Use ascontiguousarray to ensure C-contiguous memory layout for soundfile
        sf.write(chunk_dir / "mix.wav", np.ascontiguousarray(mix.T), sr, subtype="PCM_16")
        sf.write(chunk_dir / "vocals.wav", np.ascontiguousarray(vocals.T), sr, subtype="PCM_16")
        sf.write(chunk_dir / "violin.wav", np.ascontiguousarray(violin.T), sr, subtype="PCM_16")
        sf.write(chunk_dir / "mridangam.wav", np.ascontiguousarray(mridangam.T), sr, subtype="PCM_16")
        sf.write(chunk_dir / "ghatam.wav", np.ascontiguousarray(ghatam.T), sr, subtype="PCM_16")

        has_ghatam = 1 if (has_ghatam_file and float(np.dot(ghatam.ravel(), ghatam.ravel())) > 1e-12) else 0

        chunks_base = Path("chunks")
        chunks_meta.append({
            "chunk_id": chunk_id,
            "concert_id": record.concert_id,
            "song_id": record.song_id,
            "start_sec": float(start_sec),
            "duration_sec": float(chunk_duration_sec),
            "mix_path": str(chunks_base / chunk_id / "mix.wav").replace("\\", "/"),
            "vocals_path": str(chunks_base / chunk_id / "vocals.wav").replace("\\", "/"),
            "violin_path": str(chunks_base / chunk_id / "violin.wav").replace("\\", "/"),
            "mridangam_path": str(chunks_base / chunk_id / "mridangam.wav").replace("\\", "/"),
            "ghatam_path": str(chunks_base / chunk_id / "ghatam.wav").replace("\\", "/"),
            "has_ghatam": has_ghatam,
        })
        start_sec += chunk_duration_sec
        window_idx += 1

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chunk_id", "concert_id", "song_id", "start_sec",
            "duration_sec", "mix_path", "vocals_path", "violin_path",
            "mridangam_path", "ghatam_path", "has_ghatam",
        ])
        for row in chunks_meta:
            writer.writerow([
                row["chunk_id"], row["concert_id"], row["song_id"],
                f"{row['start_sec']:.1f}", f"{row['duration_sec']:.1f}",
                row["mix_path"], row["vocals_path"], row["violin_path"],
                row["mridangam_path"], row["ghatam_path"], row["has_ghatam"],
            ])

    # Console summary with skipped count
    print(f"Staged {len(chunks_meta)} chunks from {record.song_id} (C{record.concert_id:02d})")
    if skipped_silent > 0:
        print(f"  Skipped {skipped_silent} silent windows (energy <= 1e-12)")

    return chunks_meta


def stage_dataset(
    dataset_root: Path,
    output_dir: Path,
    train_concerts: tuple[int, ...] = (1, 2, 4, 5),
    chunk_duration_sec: float = 6.0,
    max_songs: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = index_sanidha(dataset_root)
    train, _ = split_sanidha(records, train_concerts=train_concerts, val_concerts=(3,))

    # Bar Concert 3 guard
    if 3 in train_concerts:
        raise ValueError("Concert 3 is the validation set and cannot be staged for training.")

    manifest_records: list[dict[str, Any]] = []
    total_chunks = 0

    for rec in train:
        if max_songs is not None and total_chunks >= max_songs:
            break
        chunks_meta = stage_song(
            rec, output_dir,
            chunk_duration_sec=chunk_duration_sec,
            include_tanpura=True,
        )
        manifest_records.extend(chunks_meta)
        total_chunks = len(manifest_records)

    # Write manifest once at the end
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chunk_id", "concert_id", "song_id", "start_sec",
            "duration_sec", "mix_path", "vocals_path", "violin_path",
            "mridangam_path", "ghatam_path", "has_ghatam",
        ])
        for row in manifest_records:
            writer.writerow([
                row["chunk_id"], row["concert_id"], row["song_id"],
                f"{row['start_sec']:.1f}", f"{row['duration_sec']:.1f}",
                row["mix_path"], row["vocals_path"], row["violin_path"],
                row["mridangam_path"], row["ghatam_path"], row["has_ghatam"],
            ])

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"C:\Users\Samar\OneDrive\Documents\Sanidha_Dataset")
    parser.add_argument("--output-dir", default="data/staged")
    parser.add_argument("--chunk-duration", type=float, default=6.0)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--train-concerts", nargs="+", type=int, default=[1, 4, 5])
    args = parser.parse_args()
    if 3 in args.train_concerts:
        raise ValueError("Concert 3 is the validation set and cannot be staged for training.")
    stage_dataset(
        Path(args.dataset_root),
        Path(args.output_dir),
        train_concerts=tuple(args.train_concerts),
        chunk_duration_sec=args.chunk_duration,
        max_songs=args.max_songs,
    )


if __name__ == "__main__":
    main()
