"""Sanidha staged dataset for 2-stem fine-tuning."""
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Optional
import pandas as pd
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SAMPLE_RATE = 44100
CHUNK_DURATION_SEC = 6.0
EXPECTED_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)  # 264600
REQUIRED_COLS = ("mix_path", "vocals_path", "accompaniment_path")


def _read_audio_ch_first(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read WAV as (channels, samples) float32; verify sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != target_sr:
        raise ValueError(f"{path}: sample rate {sr} != {target_sr}")
    # soundfile returns (samples, channels) -> (channels, samples)
    data = data.T.astype(np.float32, copy=False)
    return data


class SanidhaStagedDataset(Dataset):
    """Torch Dataset for Sanidha 2-stem staged chunks.

    Each item returns:
        mix_tensor: (2, N) float32
        targets_tensor: (2, 2, N) float32  # [vocals, accompaniment] each (2, N)
    """
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: Optional[str | Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        missing = [c for c in REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        mix_path = self.root_dir / row["mix_path"]
        vocals_path = self.root_dir / row["vocals_path"]
        accomp_path = self.root_dir / row["accompaniment_path"]

        mix = _read_audio_ch_first(mix_path)
        vocals = _read_audio_ch_first(vocals_path)
        accomp = _read_audio_ch_first(accomp_path)

        # Verify expected length (6s @ 44100 = 264600)
        for name, arr in [("mix", mix), ("vocals", vocals), ("accomp", accomp)]:
            if arr.shape != (2, EXPECTED_SAMPLES):
                raise ValueError(f"{name}: expected (2, {EXPECTED_SAMPLES}), got {arr.shape}")

        mix_t = torch.from_numpy(mix)
        vocals_t = torch.from_numpy(vocals)
        accomp_t = torch.from_numpy(accomp)

        targets_t = torch.stack([vocals_t, accomp_t], dim=0)  # (2, 2, N)
        return mix_t, targets_t