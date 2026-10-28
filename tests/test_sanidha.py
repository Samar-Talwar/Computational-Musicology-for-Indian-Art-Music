"""Test suite for utils/sanidha.py (Strict TDD - synthetic float32 fixtures).

All tests write to tmp_path via soundfile.write; never touch OneDrive dataset.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _rng(seed=42):
    return np.random.default_rng(seed)


@pytest.fixture
def sanidha_tree(tmp_path):
    """Create a mini Sanidha tree under tmp_path.

    Structure:
      tmp_path/
        Concert01/
          01-SongA/
            Audio-Multitracks-Clean/
              vocals.wav, violin-1.wav, violin-2.wav, mridangam-left.wav, mridangam-right.wav,
              ghatam-close.wav, ghatam-far.wav, tanpura.wav
          02-SongB/  (missing ghatam)
            Audio-Multitracks-Clean/
              vocals.wav, violin-1.wav, violin-2.wav, mridangam-left.wav, mridangam-right.wav,
              tanpura.wav
        Concert03/
          03-SongC/
            Audio-Multitracks-Clean/
              vocals.wav, violin-1.wav, violin-2.wav, mridangam-left.wav, mridangam-right.wav,
              ghatam-close.wav, ghatam-far.wav, tanpura.wav
    All files: stereo (2, 44100), float32, PCM_16 written via soundfile.
    """
    import soundfile as sf

    sr = 44100
    n_samples = sr  # 1 second

    def write_stem(base, name, data):
        path = base / f"{name}.wav"
        sf.write(path, data.T, sr, subtype="PCM_16")  # soundfile expects (N, C)

    for concert_name, songs in [
        ("Concert01", ["01-SongA", "02-SongB"]),
        ("Concert03", ["03-SongC"]),
    ]:
        for song in songs:
            clean_dir = tmp_path / concert_name / song / "Audio-Multitracks-Clean"
            clean_dir.mkdir(parents=True)

            # Common stems
            rng = _rng(hash(song) % 2**32)
            vocals = rng.standard_normal((2, n_samples), dtype=np.float32)
            violin1 = rng.standard_normal((2, n_samples), dtype=np.float32)
            violin2 = rng.standard_normal((2, n_samples), dtype=np.float32)
            mrid_left = rng.standard_normal((2, n_samples), dtype=np.float32)
            mrid_right = rng.standard_normal((2, n_samples), dtype=np.float32)
            tanpura = rng.standard_normal((2, n_samples), dtype=np.float32)

            write_stem(clean_dir, "vocals", vocals)
            write_stem(clean_dir, "violin-1", violin1)
            write_stem(clean_dir, "violin-2", violin2)
            write_stem(clean_dir, "mridangam-left", mrid_left)
            write_stem(clean_dir, "mridangam-right", mrid_right)
            write_stem(clean_dir, "tanpura", tanpura)

            # Ghatam only for SongA and SongC
            if song in ("01-SongA", "03-SongC"):
                ghatam_close = rng.standard_normal((2, n_samples), dtype=np.float32)
                ghatam_far = rng.standard_normal((2, n_samples), dtype=np.float32)
                write_stem(clean_dir, "ghatam-close", ghatam_close)
                write_stem(clean_dir, "ghatam-far", ghatam_far)

    return tmp_path


@pytest.fixture
def partial_violin_tree(tmp_path):
    """Tree with incomplete violin sub-mics (violin-1 present, violin-2 missing)."""
    import soundfile as sf

    sr = 44100
    n_samples = sr

    def write_stem(base, name, data):
        path = base / f"{name}.wav"
        sf.write(path, data.T, sr, subtype="PCM_16")

    concert = "Concert01"
    song = "04-SongD"
    clean_dir = tmp_path / concert / song / "Audio-Multitracks-Clean"
    clean_dir.mkdir(parents=True)

    rng = _rng(999)
    write_stem(clean_dir, "vocals", rng.standard_normal((2, n_samples), dtype=np.float32))
    write_stem(clean_dir, "violin-1", rng.standard_normal((2, n_samples), dtype=np.float32))
    # violin-2 missing
    write_stem(clean_dir, "mridangam-left", rng.standard_normal((2, n_samples), dtype=np.float32))
    write_stem(clean_dir, "mridangam-right", rng.standard_normal((2, n_samples), dtype=np.float32))
    write_stem(clean_dir, "tanpura", rng.standard_normal((2, n_samples), dtype=np.float32))

    return tmp_path


class TestSongRecordAndMap:
    def test_logical_stem_map_constant(self):
        from utils.sanidha import LOGICAL_STEM_MAP
        assert LOGICAL_STEM_MAP["vocals"] == ("vocals",)
        assert LOGICAL_STEM_MAP["violin"] == ("violin-1", "violin-2")
        assert LOGICAL_STEM_MAP["mridangam"] == ("mridangam-left", "mridangam-right")
        assert LOGICAL_STEM_MAP["ghatam"] == ("ghatam-close", "ghatam-far")
        assert LOGICAL_STEM_MAP["tanpura"] == ("tanpura",)

    def test_song_record_no_arrays(self):
        from utils.sanidha import SongRecord
        from pathlib import Path

        rec = SongRecord(
            concert_id=1,
            song_id="01-SongA",
            folder=Path("/fake/Concert01/01-SongA"),
            stem_files={"vocals": (Path("/fake/vocals.wav"),)},
        )
        # No np.ndarray in SongRecord
        assert not hasattr(rec, "stems")
        assert rec.concert_id == 1
        assert isinstance(rec.stem_files, dict)


class TestIndexSanidha:
    def test_index_finds_songs_and_populates_stem_files(self, sanidha_tree):
        from utils.sanidha import index_sanidha, LOGICAL_STEM_MAP

        records = index_sanidha(sanidha_tree)
        # Should find 3 songs
        assert len(records) == 3

        for rec in records:
            assert rec.concert_id in (1, 3)
            assert isinstance(rec.song_id, str)
            assert isinstance(rec.folder, Path)
            assert isinstance(rec.stem_files, dict)
            # vocals always present
            assert "vocals" in rec.stem_files
            assert len(rec.stem_files["vocals"]) == 1
            # violin has 2 sub-mics
            assert "violin" in rec.stem_files
            assert len(rec.stem_files["violin"]) == 2
            # mridangam has 2 sub-mics
            assert "mridangam" in rec.stem_files
            assert len(rec.stem_files["mridangam"]) == 2
            # tanpura present
            assert "tanpura" in rec.stem_files
            assert len(rec.stem_files["tanpura"]) == 1

    def test_index_handles_missing_ghatam(self, sanidha_tree):
        from utils.sanidha import index_sanidha

        records = index_sanidha(sanidha_tree)
        song_b = [r for r in records if r.song_id == "02-SongB"][0]
        # ghatam omitted entirely
        assert "ghatam" not in song_b.stem_files

    def test_index_handles_missing_tanpura(self, sanidha_tree):
        from utils.sanidha import index_sanidha

        records = index_sanidha(sanidha_tree)
        # All songs in fixture have tanpura; test ensures key present
        for rec in records:
            assert "tanpura" in rec.stem_files

    def test_index_raises_on_partial_submic(self, partial_violin_tree):
        from utils.sanidha import index_sanidha

        with pytest.raises(ValueError, match="violin.*sub-mic"):
            index_sanidha(partial_violin_tree)


class TestSplitSanidha:
    def test_split_returns_disjoint_lists(self, sanidha_tree):
        from utils.sanidha import index_sanidha, split_sanidha

        records = index_sanidha(sanidha_tree)
        train, val = split_sanidha(records)

        # train = concerts 1, 2, 4, 5; val = concert 3
        assert all(r.concert_id in (1, 2, 4, 5) for r in train)
        assert all(r.concert_id == 3 for r in val)
        assert len(train) + len(val) == len(records)
        # disjoint
        train_ids = {id(r) for r in train}
        val_ids = {id(r) for r in val}
        assert train_ids.isdisjoint(val_ids)

    def test_split_raises_on_overlap(self, sanidha_tree):
        from utils.sanidha import index_sanidha, split_sanidha

        records = index_sanidha(sanidha_tree)
        with pytest.raises(ValueError, match="overlap"):
            split_sanidha(records, train_concerts=(1, 3), val_concerts=(3,))

    def test_split_custom_concerts(self, sanidha_tree):
        from utils.sanidha import index_sanidha, split_sanidha

        records = index_sanidha(sanidha_tree)
        train, val = split_sanidha(records, train_concerts=(1,), val_concerts=(3,))
        assert all(r.concert_id == 1 for r in train)
        assert all(r.concert_id == 3 for r in val)


class TestLoadLogicalStem:
    def test_load_vocals_returns_float32_shape(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        vocals = load_logical_stem(rec, "vocals")
        assert vocals.shape == (2, 44100)
        assert vocals.dtype == np.float32

    def test_load_violin_sums_submics(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem
        import soundfile as sf

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        violin = load_logical_stem(rec, "violin")
        # Sum of violin-1 + violin-2 (rec.folder is already Audio-Multitracks-Clean)
        v1, _ = sf.read(rec.folder / "violin-1.wav", dtype="float32")
        v2, _ = sf.read(rec.folder / "violin-2.wav", dtype="float32")
        expected = (v1.T + v2.T).astype(np.float32)
        assert np.allclose(violin, expected)
        assert violin.dtype == np.float32

    def test_load_ghatam_missing_raises(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem

        records = index_sanidha(sanidha_tree)
        song_b = [r for r in records if r.song_id == "02-SongB"][0]
        with pytest.raises(ValueError, match="ghatam"):
            load_logical_stem(song_b, "ghatam")

    def test_load_sr_mismatch_raises(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem
        import soundfile as sf

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        # Corrupt one file to 22050 Hz
        bad_path = rec.folder / "violin-1.wav"
        data, _ = sf.read(bad_path, dtype="float32")
        sf.write(bad_path, data, 22050, subtype="PCM_16")
        try:
            with pytest.raises(ValueError, match="sample rate|sr"):
                load_logical_stem(rec, "violin", target_sr=44100)
        finally:
            # Restore
            sf.write(bad_path, data, 44100, subtype="PCM_16")

    def test_load_frame_length_mismatch_raises(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem
        import soundfile as sf

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        # Truncate one sub-mic
        bad_path = rec.folder / "violin-1.wav"
        data, _ = sf.read(bad_path, dtype="float32")
        sf.write(bad_path, data[:1000], 44100, subtype="PCM_16")
        try:
            with pytest.raises(ValueError, match="frame|length|shape"):
                load_logical_stem(rec, "violin", target_sr=44100)
        finally:
            sf.write(bad_path, data, 44100, subtype="PCM_16")

    def test_load_time_window(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_logical_stem

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        # Load 0.5 sec starting at 0.25 sec
        seg = load_logical_stem(rec, "vocals", start_sec=0.25, duration_sec=0.5)
        assert seg.shape == (2, 22050)  # 0.5 * 44100
        assert seg.dtype == np.float32


class TestLoadSong:
    def test_load_song_returns_mix_and_stems(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_song

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        mix, stems = load_song(rec)

        assert mix.shape == (2, 44100)
        assert mix.dtype == np.float32
        assert set(stems.keys()) >= {"vocals", "violin", "mridangam", "tanpura"}
        # mix == sum of stems
        expected_mix = sum(stems.values())
        assert np.allclose(mix, expected_mix)
        assert mix.dtype == np.float32

    def test_load_song_excludes_tanpura(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_song

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        mix, stems = load_song(rec, include_tanpura=False)

        assert "tanpura" not in stems
        expected = sum(v for k, v in stems.items() if k != "tanpura")
        assert np.allclose(mix, expected)

    def test_load_song_compatible_with_collapse_to_two_stem(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_song
        from evaluation.metrics import collapse_to_two_stem

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        _, stems = load_song(rec)
        # Should be able to collapse
        two = collapse_to_two_stem(stems, vocal_key="vocals")
        assert set(two.keys()) == {"vocals", "accompaniment"}
        assert two["vocals"].shape == (2, 44100)
        assert two["accompaniment"].shape == (2, 44100)

    def test_load_song_raises_on_nonfinite(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_song
        import soundfile as sf

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        # Create a temp file with IEEE_FLOAT subtype (supports NaN/Inf)
        bad_path = rec.folder / "vocals_nan.wav"
        # Create vocals with NaN
        bad_data = np.zeros((44100, 2), dtype=np.float32)  # (N, C) for soundfile
        bad_data[0, 0] = np.nan
        sf.write(bad_path, bad_data, 44100, subtype="FLOAT")

        # Temporarily patch stem_files to point to our bad file
        original_stem_files = rec.stem_files
        # SongRecord is frozen, so create a new one with modified stem_files
        from utils.sanidha import SongRecord
        bad_rec = SongRecord(
            concert_id=rec.concert_id,
            song_id=rec.song_id,
            folder=rec.folder,
            stem_files={**original_stem_files, "vocals": (bad_path,)},
        )
        try:
            with pytest.raises(ValueError, match="finite"):
                load_song(bad_rec)
        finally:
            bad_path.unlink()

    def test_load_song_raises_on_mismatched_shapes(self, sanidha_tree):
        from utils.sanidha import index_sanidha, load_song
        import soundfile as sf

        records = index_sanidha(sanidha_tree)
        rec = records[0]
        # Make violin shorter
        bad_path = rec.folder / "violin-1.wav"
        data, _ = sf.read(bad_path, dtype="float32")  # (N, C)
        sf.write(bad_path, data[:1000], 44100, subtype="PCM_16")
        try:
            with pytest.raises(ValueError, match="shape|mismatch"):
                load_song(rec)
        finally:
            sf.write(bad_path, data, 44100, subtype="PCM_16")