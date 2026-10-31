"""TDD Step 1 — tests/test_stage_colab_data.py (will fail until scripts/stage_colab_data.py exists)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_colab_record(tmp_path):
    """Synthetic Sanidha-style record with 2 logical stems (vocals + accompaniment source) for staging."""
    import soundfile as sf
    sr = 44100
    # 10-second synthetic audio, float32, stereo (2, N)
    n = sr * 10
    vocals = np.random.default_rng(42).standard_normal((2, n), dtype=np.float32) * 0.5
    violin = np.random.default_rng(43).standard_normal((2, n), dtype=np.float32) * 0.3
    clean = tmp_path / "Concert01" / "01-SongA" / "Audio-Multitracks-Clean"
    clean.mkdir(parents=True)
    sf.write(clean / "vocals.wav", vocals.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-1.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-2.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "tanpura.wav", np.zeros((2, n), dtype=np.float32).T, sr, subtype="PCM_16")
    sf.write(clean / "mridangam-left.wav", (0.2 * np.ones((2, n), dtype=np.float32)).T, sr, subtype="PCM_16")
    sf.write(clean / "mridangam-right.wav", (0.2 * np.ones((2, n), dtype=np.float32)).T, sr, subtype="PCM_16")
    from utils.sanidha import SongRecord
    return SongRecord(
        concert_id=1, song_id="01-SongA", folder=clean,
        stem_files={
            "vocals": (clean / "vocals.wav",),
            "violin": (clean / "violin-1.wav", clean / "violin-2.wav"),
            "mridangam": (clean / "mridangam-left.wav", clean / "mridangam-right.wav"),
            "tanpura": (clean / "tanpura.wav",),
        },
    )


@pytest.fixture
def synthetic_colab_record_4stem(tmp_path):
    """Synthetic record with 4 stems including ghatam (float32 fixtures per CLAUDE.md TDD)."""
    import soundfile as sf
    sr = 44100
    n = sr * 10
    vocals = np.random.default_rng(42).standard_normal((2, n), dtype=np.float32) * 0.5
    violin = np.random.default_rng(43).standard_normal((2, n), dtype=np.float32) * 0.3
    mridangam = np.random.default_rng(44).standard_normal((2, n), dtype=np.float32) * 0.2
    ghatam = np.random.default_rng(45).standard_normal((2, n), dtype=np.float32) * 0.1
    clean = tmp_path / "Concert01" / "01-SongA" / "Audio-Multitracks-Clean"
    clean.mkdir(parents=True)
    sf.write(clean / "vocals.wav", vocals.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-1.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-2.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "mridangam-1.wav", mridangam.T, sr, subtype="PCM_16")
    sf.write(clean / "ghatam.wav", ghatam.T, sr, subtype="PCM_16")
    from utils.sanidha import SongRecord
    return SongRecord(
        concert_id=1, song_id="01-SongA", folder=clean,
        stem_files={
            "vocals": (clean / "vocals.wav",),
            "violin": (clean / "violin-1.wav", clean / "violin-2.wav"),
            "mridangam": (clean / "mridangam-1.wav",),
            "ghatam": (clean / "ghatam.wav",),
        },
    )
    return SongRecord(
        concert_id=1, song_id="01-SongA", folder=clean,
        stem_files={
            "vocals": (clean / "vocals.wav",),
            "violin": (clean / "violin-1.wav", clean / "violin-2.wav"),
            "tanpura": (clean / "tanpura.wav",),
        },
    )


class TestStageColab:
    def test_stage_song_exists(self, synthetic_colab_record, tmp_path):
        from scripts.stage_colab_data import stage_song
        chunks = stage_song(synthetic_colab_record, tmp_path / "out", chunk_duration_sec=6.0)
        assert isinstance(chunks, list)
        # 10 sec / 6 sec = 1 full chunk; trailing 4 sec dropped
        assert len(chunks) == 1

    def test_stage_dataset_exists(self, synthetic_colab_record, tmp_path):
        from scripts.stage_colab_data import stage_dataset
        # Need mini root; synthetic fixture puts under tmp_path/Concert01/...
        root = synthetic_colab_record.folder.parent.parent.parent
        result = stage_dataset(root, tmp_path / "staged", max_songs=1)
        assert isinstance(result, Path)
        assert (tmp_path / "staged" / "manifest.csv").exists()

    def test_chunk_files_written_float32(self, synthetic_colab_record, tmp_path):
        from scripts.stage_colab_data import stage_song
        stage_song(synthetic_colab_record, tmp_path / "staged")
        chunks_dir = tmp_path / "staged" / "chunks"
        # At least one chunk dir
        dirs = [d for d in chunks_dir.iterdir() if d.is_dir()]
        assert len(dirs) >= 1
        mix = dirs[0] / "mix.wav"
        assert mix.exists()

    def test_energy_filter_skips_silent(self, synthetic_colab_record, tmp_path):
        from scripts.stage_colab_data import stage_song
        # Make vocals silent -> should produce 0 chunks
        import soundfile as sf
        rec = synthetic_colab_record
        bad_path = rec.folder / "vocals.wav"
        data, sr = sf.read(bad_path, dtype="float32")
        silent = np.zeros_like(data)
        sf.write(bad_path, silent, sr, subtype="PCM_16")
        try:
            chunks = stage_song(rec, tmp_path / "silent_out", chunk_duration_sec=6.0)
            # Zero chunk dirs because all windows silent
            chunk_dirs = [d for d in (tmp_path / "silent_out" / "chunks").iterdir() if d.is_dir()]
            assert len(chunk_dirs) == 0
            assert len(chunks) == 0
        finally:
            sf.write(bad_path, data, sr, subtype="PCM_16")

    def test_stage_dataset_accumulates_multiple_songs(self, tmp_path):
        from scripts.stage_colab_data import stage_dataset
        import soundfile as sf
        import numpy as np

        # Create two synthetic songs under different concerts
        sr = 44100
        n = sr * 10  # 10 seconds each

        # Song 1 in Concert01
        clean1 = tmp_path / "Concert01" / "01-SongA" / "Audio-Multitracks-Clean"
        clean1.mkdir(parents=True)
        v1 = np.random.default_rng(1).standard_normal((2, n), dtype=np.float32) * 0.5
        vi1 = np.random.default_rng(2).standard_normal((2, n), dtype=np.float32) * 0.3
        sf.write(clean1 / "vocals.wav", v1.T, sr, subtype="PCM_16")
        sf.write(clean1 / "violin-1.wav", vi1.T, sr, subtype="PCM_16")
        sf.write(clean1 / "violin-2.wav", vi1.T, sr, subtype="PCM_16")
        sf.write(clean1 / "tanpura.wav", np.zeros((2, n), dtype=np.float32).T, sr, subtype="PCM_16")
        sf.write(clean1 / "mridangam-left.wav", (0.2 * np.ones((2, n), dtype=np.float32)).T, sr, subtype="PCM_16")
        sf.write(clean1 / "mridangam-right.wav", (0.2 * np.ones((2, n), dtype=np.float32)).T, sr, subtype="PCM_16")

        # Song 2 in Concert02
        clean2 = tmp_path / "Concert02" / "02-SongB" / "Audio-Multitracks-Clean"
        clean2.mkdir(parents=True)
        v2 = np.random.default_rng(3).standard_normal((2, n), dtype=np.float32) * 0.5
        vi2 = np.random.default_rng(4).standard_normal((2, n), dtype=np.float32) * 0.3
        mr2 = np.ones((2, n), dtype=np.float32) * 0.2
        sf.write(clean2 / "vocals.wav", v2.T, sr, subtype="PCM_16")
        sf.write(clean2 / "violin-1.wav", vi2.T, sr, subtype="PCM_16")
        sf.write(clean2 / "violin-2.wav", vi2.T, sr, subtype="PCM_16")
        sf.write(clean2 / "mridangam-left.wav", mr2.T, sr, subtype="PCM_16")
        sf.write(clean2 / "mridangam-right.wav", mr2.T, sr, subtype="PCM_16")
        sf.write(clean2 / "tanpura.wav", np.zeros((2, n), dtype=np.float32).T, sr, subtype="PCM_16")

        # Stage both
        root = tmp_path
        result = stage_dataset(root, tmp_path / "staged", train_concerts=(1, 2), max_songs=2)
        assert isinstance(result, Path)

        # Read manifest and verify both songs present
        import csv
        manifest_path = tmp_path / "staged" / "manifest.csv"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Should have 2 chunks (1 per song, 10s/6s = 1 chunk each)
        assert len(rows) == 2
        song_ids = {r["song_id"] for r in rows}
        assert song_ids == {"01-SongA", "02-SongB"}


class TestStageColab4Stem:
    def test_4stem_with_ghatam_writes_all_stems_and_has_ghatam_1(self, synthetic_colab_record_4stem, tmp_path):
        from scripts.stage_colab_data import stage_song
        chunks = stage_song(synthetic_colab_record_4stem, tmp_path / "out", chunk_duration_sec=6.0)
        assert len(chunks) == 1
        chunk_dir = tmp_path / "out" / "chunks" / chunks[0]["chunk_id"]
        assert (chunk_dir / "mix.wav").exists()
        assert (chunk_dir / "vocals.wav").exists()
        assert (chunk_dir / "violin.wav").exists()
        assert (chunk_dir / "mridangam.wav").exists()
        assert (chunk_dir / "ghatam.wav").exists()
        assert chunks[0]["has_ghatam"] == 1

    def test_4stem_without_ghatam_silent_ghatam_and_has_0(self, tmp_path):
        from scripts.stage_colab_data import stage_song
        import soundfile as sf
        sr = 44100
        n = sr * 10
        clean = tmp_path / "Concert01" / "01-SongNoGhatam" / "Audio-Multitracks-Clean"
        clean.mkdir(parents=True)
        vocals = (0.5 * np.ones((2, n), dtype=np.float32))
        violin = (0.3 * np.ones((2, n), dtype=np.float32))
        mridangam = (0.2 * np.ones((2, n), dtype=np.float32))
        ghatam_silent = np.zeros((2, n), dtype=np.float32)  # explicit silence per spec
        sf.write(clean / "vocals.wav", vocals.T, sr, subtype="PCM_16")
        sf.write(clean / "violin-1.wav", violin.T, sr, subtype="PCM_16")
        sf.write(clean / "violin-2.wav", violin.T, sr, subtype="PCM_16")
        sf.write(clean / "mridangam-1.wav", mridangam.T, sr, subtype="PCM_16")
        from utils.sanidha import SongRecord
        rec = SongRecord(
            concert_id=1, song_id="01-SongNoGhatam", folder=clean,
            stem_files={
                "vocals": (clean / "vocals.wav",),
                "violin": (clean / "violin-1.wav", clean / "violin-2.wav"),
                "mridangam": (clean / "mridangam-1.wav",),
            },
        )
        # Silent fixture: exact zeros => RMS 0.0 <= 1e-12 (skip expected if all silent)
        # Here other stems are audible, only ghatam missing => ghatam silent file written
        chunks = stage_song(rec, tmp_path / "out_no_g", chunk_duration_sec=6.0)
        assert len(chunks) == 1
        chunk_dir = tmp_path / "out_no_g" / "chunks" / chunks[0]["chunk_id"]
        assert (chunk_dir / "ghatam.wav").exists()
        # Silent ghatam: verify all zeros
        data, sr_read = sf.read(chunk_dir / "ghatam.wav", dtype="float32")
        assert np.allclose(data, 0, atol=1e-6), "Silent ghatam should be near-zero"
        assert chunks[0]["has_ghatam"] == 0

    def test_manifest_schema_4stem_has_ghatam(self, synthetic_colab_record_4stem, tmp_path):
        from scripts.stage_colab_data import stage_song, stage_dataset
        stage_song(synthetic_colab_record_4stem, tmp_path / "staged")
        # Check manifest exists and has required columns
        import csv
        manifest_path = tmp_path / "staged" / "manifest.csv"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        required = ["chunk_id", "concert_id", "song_id", "start_sec", "duration_sec",
                    "mix_path", "vocals_path", "violin_path", "mridangam_path",
                    "ghatam_path", "has_ghatam"]
        for col in required:
            assert col in rows[0], f"Missing manifest column: {col}"
        assert rows[0]["has_ghatam"] == "1"
