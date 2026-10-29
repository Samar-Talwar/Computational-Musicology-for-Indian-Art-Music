"""TDD Step 1 — tests/test_dataset.py (will fail until models/dataset.py exists)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_manifest(tmp_path):
    """Create a minimal staged manifest + chunk WAVs for testing."""
    import soundfile as sf
    import csv

    sr = 44100
    n = sr * 6  # 6 seconds = 264600
    root = tmp_path / "staged"
    chunks_dir = root / "chunks"
    chunks_dir.mkdir(parents=True)

    # Create 3 synthetic chunks
    manifest_rows = []
    for i in range(3):
        chunk_id = f"C01_01-SongA_chunk{i:04d}"
        chunk_dir = chunks_dir / chunk_id
        chunk_dir.mkdir(parents=True)

        # Generate float32 stereo (2, N) scaled to [-1, 1] for PCM_16
        rng = np.random.default_rng(100 + i)
        # Generate base signals in [-1, 1] range
        base = rng.uniform(-0.5, 0.5, (2, n)).astype(np.float32)
        vocals = rng.uniform(-0.5, 0.5, (2, n)).astype(np.float32) * 0.3
        mix = base + vocals
        accomp = base  # mix - vocals = base

        # Write as (N, 2) for soundfile
        sf.write(chunk_dir / "mix.wav", mix.T, sr, subtype="PCM_16")
        sf.write(chunk_dir / "vocals.wav", vocals.T, sr, subtype="PCM_16")
        sf.write(chunk_dir / "accompaniment.wav", accomp.T, sr, subtype="PCM_16")

        manifest_rows.append({
            "chunk_id": chunk_id,
            "concert_id": 1,
            "song_id": "01-SongA",
            "start_sec": float(i * 6.0),
            "duration_sec": 6.0,
            "mix_path": f"chunks/{chunk_id}/mix.wav",
            "vocals_path": f"chunks/{chunk_id}/vocals.wav",
            "accompaniment_path": f"chunks/{chunk_id}/accompaniment.wav",
        })

    # Write manifest.csv
    manifest_path = root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path, root


class TestSanidhaStagedDataset:
    def test_dataset_len(self, synthetic_manifest):
        from models.dataset import SanidhaStagedDataset
        manifest_path, root = synthetic_manifest
        ds = SanidhaStagedDataset(manifest_path, root)
        assert len(ds) == 3

    def test_getitem_shapes_and_dtype(self, synthetic_manifest):
        from models.dataset import SanidhaStagedDataset
        import torch
        manifest_path, root = synthetic_manifest
        ds = SanidhaStagedDataset(manifest_path, root)
        mix, targets = ds[0]
        assert isinstance(mix, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert mix.shape == (2, 264600)
        assert targets.shape == (2, 2, 264600)
        assert mix.dtype == torch.float32
        assert targets.dtype == torch.float32

    def test_targets_order_vocals_accomp(self, synthetic_manifest):
        from models.dataset import SanidhaStagedDataset
        import torch
        manifest_path, root = synthetic_manifest
        ds = SanidhaStagedDataset(manifest_path, root)
        mix, targets = ds[0]
        # targets[0] = vocals, targets[1] = accompaniment
        assert targets.shape[0] == 2
        # vocals + accomp should approximately equal mix (float precision)
        recon = targets[0] + targets[1]
        assert torch.allclose(recon, mix, atol=1e-3, rtol=1e-3)

    def test_missing_column_raises(self, tmp_path):
        from models.dataset import SanidhaStagedDataset
        import pandas as pd
        # Manifest missing required columns
        bad_manifest = tmp_path / "bad_manifest.csv"
        pd.DataFrame({"chunk_id": ["a"], "mix_path": ["x"], "vocals_path": ["y"]}).to_csv(bad_manifest, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            SanidhaStagedDataset(bad_manifest, tmp_path)

    def test_default_root_dir_from_manifest_parent(self, synthetic_manifest):
        from models.dataset import SanidhaStagedDataset
        manifest_path, root = synthetic_manifest
        ds = SanidhaStagedDataset(manifest_path)  # no root_dir -> defaults to manifest.parent
        assert len(ds) == 3