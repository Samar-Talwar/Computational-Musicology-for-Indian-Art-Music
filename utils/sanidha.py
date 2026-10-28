"""Sanidha dataset utilities - zero-bleed Carnatic multi-track loader.

All functions use explicit contracts, no silent defaults.
Paths only in SongRecord; waveforms loaded on demand.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import soundfile as sf


LOGICAL_STEM_MAP = {
    "vocals": ("vocals",),
    "violin": ("violin-1", "violin-2"),
    "mridangam": ("mridangam-left", "mridangam-right"),
    "ghatam": ("ghatam-close", "ghatam-far"),
    "tanpura": ("tanpura",),
}


@dataclass(frozen=True)
class SongRecord:
    """Metadata for a single song in Sanidha.

    Attributes:
        concert_id: 1-5 (from Concert01 etc.)
        song_id: folder name (e.g. "01-Vanajakshi-Varnam")
        folder: Path to ConcertXX/<song_id>/
        stem_files: logical stem name -> tuple of sub-mic .wav Paths on disk.
                    Absent stems (e.g. ghatam in C02/C04/C05, or tanpura in missing songs)
                    are simply omitted from this dict.
    """
    concert_id: int
    song_id: str
    folder: Path
    stem_files: dict[str, tuple[Path, ...]]


def index_sanidha(root: str | Path) -> list[SongRecord]:
    """Scan Sanidha root and return SongRecord list (Clean folder only).

    Args:
        root: Path to Sanidha root containing Concert01..Concert05.

    Returns:
        List of SongRecord, one per song found under Clean subfolder.

    Raises:
        ValueError: if a multi-mic stem (violin, mridangam, ghatam) has only
                    1 of its required sub-mic files present.
    """
    root = Path(root)
    records = []

    for concert_dir in sorted(root.iterdir()):
        if not concert_dir.is_dir() or not concert_dir.name.startswith("Concert"):
            continue
        # Parse concert number from "Concert01" -> 1
        try:
            concert_id = int(concert_dir.name.replace("Concert", ""))
        except ValueError:
            continue

        for song_dir in sorted(concert_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            clean_dir = song_dir / "Audio-Multitracks-Clean"
            if not clean_dir.is_dir():
                continue

            song_id = song_dir.name
            stem_files = {}

            # Build stem_files by checking each logical stem's sub-mics
            for logical, sub_mics in LOGICAL_STEM_MAP.items():
                found_paths = []
                for sub in sub_mics:
                    path = clean_dir / f"{sub}.wav"
                    if path.is_file():
                        found_paths.append(path)

                if not found_paths:
                    # Stem absent entirely (e.g., ghatam in C02, C04, C05)
                    continue

                if len(found_paths) != len(sub_mics):
                    # Partial sub-mic presence = error
                    raise ValueError(
                        f"Song {concert_dir.name}/{song_id}: logical stem '{logical}' "
                        f"requires {len(sub_mics)} sub-mics ({sub_mics}), found {len(found_paths)}"
                    )

                stem_files[logical] = tuple(found_paths)

            if not stem_files:
                continue

            records.append(SongRecord(
                concert_id=concert_id,
                song_id=song_id,
                folder=clean_dir,
                stem_files=stem_files,
            ))

    return records


def split_sanidha(
    records: list[SongRecord],
    train_concerts: tuple[int, ...] = (1, 2, 4, 5),
    val_concerts: tuple[int, ...] = (3,),
) -> tuple[list[SongRecord], list[SongRecord]]:
    """Split records into train/val by concert_id.

    Args:
        records: Output of index_sanidha.
        train_concerts: Concert IDs for training.
        val_concerts: Concert IDs for validation.

    Returns:
        (train_records, val_records) — disjoint lists.

    Raises:
        ValueError: if train_concerts and val_concerts overlap.
    """
    train_set = set(train_concerts)
    val_set = set(val_concerts)

    if train_set & val_set:
        raise ValueError(
            f"train_concerts and val_concerts overlap: {train_set & val_set}"
        )

    train = [r for r in records if r.concert_id in train_set]
    val = [r for r in records if r.concert_id in val_set]

    return train, val


def _load_submics_sum(paths: tuple[Path, ...], target_sr: int = 44100,
                       start_sec: float = 0.0, duration_sec: float | None = None) -> np.ndarray:
    """Load multiple sub-mic files, verify sr/shape, return sum as (C, N) float32."""
    if not paths:
        raise ValueError("no paths provided")

    # Read first file to get reference sr and shape
    data0, sr0 = sf.read(paths[0], dtype="float32", always_2d=True)
    if sr0 != target_sr:
        raise ValueError(f"sample rate mismatch: {sr0} != {target_sr}")

    # Convert to (channels, samples)
    data0 = data0.T  # soundfile returns (N, C)
    n_channels, n_samples = data0.shape

    # Calculate slice
    start_frame = int(start_sec * target_sr)
    if duration_sec is not None:
        end_frame = min(start_frame + int(duration_sec * target_sr), n_samples)
    else:
        end_frame = n_samples
    start_frame = max(0, min(start_frame, n_samples))
    end_frame = max(start_frame, end_frame)

    # Sum all sub-mics
    total = np.zeros((n_channels, end_frame - start_frame), dtype=np.float32)

    for path in paths:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != target_sr:
            raise ValueError(f"{path}: sample rate {sr} != target {target_sr}")
        data = data.T
        if data.shape != (n_channels, n_samples):
            raise ValueError(
                f"{path}: shape {data.shape} != reference {(n_channels, n_samples)}"
            )
        total += data[:, start_frame:end_frame]

    return total.astype(np.float32, copy=False)


def load_logical_stem(
    record: SongRecord,
    stem_name: str,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_sr: int = 44100,
) -> np.ndarray:
    """Load a logical stem (sum of its sub-mics) from a SongRecord.

    Args:
        record: SongRecord from index_sanidha.
        stem_name: One of "vocals", "violin", "mridangam", "ghatam", "tanpura".
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds (None = to end).
        target_sr: Expected sample rate.

    Returns:
        np.ndarray shape (channels, samples), float32.

    Raises:
        ValueError: if stem_name not in record.stem_files, or sr/shape mismatch.
    """
    if stem_name not in record.stem_files:
        raise ValueError(
            f"Stem '{stem_name}' not found in {record.concert_id}/{record.song_id}. "
            f"Available: {sorted(record.stem_files.keys())}"
        )
    return _load_submics_sum(
        record.stem_files[stem_name],
        target_sr=target_sr,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )


def load_song(
    record: SongRecord,
    *,
    include_tanpura: bool = True,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_sr: int = 44100,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load all present logical stems for a song and construct mix.

    Args:
        record: SongRecord from index_sanidha.
        include_tanpura: If False, omit tanpura from stems and mix.
        start_sec, duration_sec, target_sr: Passed to load_logical_stem.

    Returns:
        (mix, stems) where:
        - mix: np.ndarray (channels, samples) float32, sum of all stems.
        - stems: dict[str, np.ndarray] float32, each (channels, samples).

    Raises:
        ValueError: on non-finite values or mismatched stem shapes.
    """
    stems = {}
    reference_shape = None

    for stem_name in record.stem_files:
        if stem_name == "tanpura" and not include_tanpura:
            continue
        stem_data = load_logical_stem(
            record, stem_name,
            start_sec=start_sec,
            duration_sec=duration_sec,
            target_sr=target_sr,
        )
        if not np.all(np.isfinite(stem_data)):
            raise ValueError(f"stem '{stem_name}' contains non-finite values")
        if reference_shape is None:
            reference_shape = stem_data.shape
        elif stem_data.shape != reference_shape:
            raise ValueError(
                f"stem '{stem_name}' shape {stem_data.shape} != "
                f"reference {reference_shape}"
            )
        stems[stem_name] = stem_data

    # Construct mix = sum of stems
    mix = sum(stems.values())

    return mix.astype(np.float32, copy=False), stems