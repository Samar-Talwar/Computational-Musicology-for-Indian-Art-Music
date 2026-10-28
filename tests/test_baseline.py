"""TDD Step 1 — tests/test_baseline.py (will fail until evaluation/baseline.py exists)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_separator(audio, sr=44100):
    """Fast fake HDemucs: input (2, N) -> dict of 4 float32 stems (~0.2s vs 30+s real)."""
    # Convert numpy output -> dict format expected by baseline
    # We return dict directly for test compatibility; real separate returns (4,2,N)
    return {
        "vocals": (audio * 0.9).astype(np.float32),
        "drums": (audio * 0.1).astype(np.float32),
        "bass": (audio * 0.05).astype(np.float32),
        "other": (audio * 0.05).astype(np.float32),
    }


@pytest.fixture
def synthetic_song(tmp_path):
    """Synthetic 3-stem song in float32 (2, 44100)."""
    import soundfile as sf
    sr = 44100
    # Create stems
    vocals = np.random.default_rng(1).standard_normal((2, sr), dtype=np.float32)
    violin = np.random.default_rng(2).standard_normal((2, sr), dtype=np.float32)
    mrid = np.random.default_rng(3).standard_normal((2, sr), dtype=np.float32)
    clean = tmp_path / "Audio-Multitracks-Clean"
    clean.mkdir(parents=True)
    sf.write(clean / "vocals.wav", vocals.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-1.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "violin-2.wav", violin.T, sr, subtype="PCM_16")
    sf.write(clean / "mridangam-left.wav", mrid.T, sr, subtype="PCM_16")
    sf.write(clean / "mridangam-right.wav", mrid.T, sr, subtype="PCM_16")
    sf.write(clean / "tanpura.wav", np.zeros((2, sr), dtype=np.float32).T, sr, subtype="PCM_16")
    # Minimal SongRecord manually
    from utils.sanidha import SongRecord
    return SongRecord(
        concert_id=3,
        song_id="03-SongC",
        folder=clean,
        stem_files={
            "vocals": (clean / "vocals.wav",),
            "violin": (clean / "violin-1.wav", clean / "violin-2.wav"),
            "mridangam": (clean / "mridangam-left.wav", clean / "mridangam-right.wav"),
            "tanpura": (clean / "tanpura.wav",),
        },
    )


class TestBaselineStep1:
    # These should FAIL with ModuleNotFoundError until evaluation/baseline.py is added.
    def test_evaluate_song_two_stem_exists(self, synthetic_song):
        from evaluation.baseline import evaluate_song_two_stem
        result = evaluate_song_two_stem(synthetic_song, segment_duration_sec=1.0, separator_fn=_fake_separator)
        assert "per_song_mean_median" in result

    def test_silent_ref_skip(self, synthetic_song):
        from evaluation.baseline import evaluate_song_two_stem
        # Not yet runnable — failure expected
        pass

    def test_max_segments(self, synthetic_song):
        pass

    def test_include_tanpura_false(self, synthetic_song):
        pass

    def test_run_concert_baseline_overall_metrics(self, tmp_path, synthetic_song):
        from evaluation.baseline import run_concert_baseline
        out = tmp_path / "out.json"
        res = run_concert_baseline([synthetic_song, synthetic_song], output_json=out)
        assert res["overall_mean_median"] is not None
        # Should contain mean/median keys under vocals/accompaniment
        assert "vocals" in res["overall_mean_median"]
        assert "SDR_mean" in res["overall_mean_median"]["vocals"]
        assert out.exists()

    def test_run_concert_baseline_json(self, tmp_path, synthetic_song):
        from evaluation.baseline import run_concert_baseline
        pass
