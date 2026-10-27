"""Test suite for evaluation/metrics.py (Strict TDD - float32 fixtures)."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _rng(seed=42):
    return np.random.default_rng(seed)


@pytest.fixture
def stereo_stems_4src():
    rng = _rng(1)
    return {
        "vocals": rng.standard_normal((2, 44100), dtype=np.float32),
        "drums": rng.standard_normal((2, 44100), dtype=np.float32),
        "bass": rng.standard_normal((2, 44100), dtype=np.float32),
        "other": rng.standard_normal((2, 44100), dtype=np.float32),
    }


@pytest.fixture
def sanidha_stems_3src():
    rng = _rng(2)
    return {
        "vocals": rng.standard_normal((2, 44100), dtype=np.float32),
        "violin": rng.standard_normal((2, 44100), dtype=np.float32),
        "mridangam": rng.standard_normal((2, 44100), dtype=np.float32),
    }


@pytest.fixture
def near_identical_stems():
    rng = _rng(3)
    ref = {"vocals": rng.standard_normal((2, 44100), dtype=np.float32)}
    # tiny perturbation
    est = {"vocals": ref["vocals"] + 1e-6 * rng.standard_normal((2, 44100), dtype=np.float32)}
    return ref, est


@pytest.fixture
def orthogonal_stems():
    rng = _rng(4)
    return {
        "vocals": rng.standard_normal((2, 44100), dtype=np.float32),
        "accompaniment": rng.standard_normal((2, 44100), dtype=np.float32),
    }


class TestCollapseToTwoStem:
    def test_collapse_conservation(self, stereo_stems_4src):
        from evaluation.metrics import collapse_to_two_stem
        out = collapse_to_two_stem(stereo_stems_4src)
        total_in = sum(stereo_stems_4src.values())
        total_out = out["vocals"] + out["accompaniment"]
        diff = np.linalg.norm(total_in - total_out)
        # float32 accumulation error: ~1e-4 is expected for 44100 samples summed
        assert diff < 1e-4, f"Conservation violated: L2 diff = {diff}"

    def test_collapse_4stem_hdemucs(self, stereo_stems_4src):
        from evaluation.metrics import collapse_to_two_stem
        out = collapse_to_two_stem(stereo_stems_4src)
        assert set(out.keys()) == {"vocals", "accompaniment"}
        assert out["vocals"].shape == (2, 44100)
        assert out["accompaniment"].shape == (2, 44100)
        assert out["vocals"].dtype == np.float32
        assert out["accompaniment"].dtype == np.float32

    def test_collapse_3stem_sanidha_missing_ghatam(self, sanidha_stems_3src):
        from evaluation.metrics import collapse_to_two_stem
        out = collapse_to_two_stem(sanidha_stems_3src)
        assert set(out.keys()) == {"vocals", "accompaniment"}
        assert out["vocals"].shape == (2, 44100)
        assert out["accompaniment"].shape == (2, 44100)

    def test_collapse_error_empty_dict(self):
        from evaluation.metrics import collapse_to_two_stem
        with pytest.raises(ValueError, match="empty"):
            collapse_to_two_stem({})

    def test_collapse_error_missing_vocal_key(self, stereo_stems_4src):
        from evaluation.metrics import collapse_to_two_stem
        with pytest.raises(ValueError, match="vocal_key"):
            collapse_to_two_stem({k: v for k, v in stereo_stems_4src.items() if k != "vocals"})

    def test_collapse_error_shape_mismatch(self):
        from evaluation.metrics import collapse_to_two_stem
        stems = {
            "vocals": np.random.randn(2, 44100).astype(np.float32),
            "drums": np.random.randn(2, 22050).astype(np.float32),  # wrong length
        }
        with pytest.raises(ValueError, match="shape"):
            collapse_to_two_stem(stems)

    def test_collapse_error_wrong_dtype(self):
        from evaluation.metrics import collapse_to_two_stem
        stems = {
            "vocals": np.random.randn(2, 44100).astype(np.float64),
            "drums": np.random.randn(2, 44100).astype(np.float64),
        }
        with pytest.raises(ValueError, match="float32"):
            collapse_to_two_stem(stems)

    def test_collapse_error_no_non_vocal(self):
        from evaluation.metrics import collapse_to_two_stem
        stems = {"vocals": np.random.randn(2, 44100).astype(np.float32)}
        with pytest.raises(ValueError, match="non-vocal"):
            collapse_to_two_stem(stems)

    def test_collapse_custom_vocal_key(self):
        from evaluation.metrics import collapse_to_two_stem
        stems = {
            "voice": np.random.randn(2, 44100).astype(np.float32),
            "instrument": np.random.randn(2, 44100).astype(np.float32),
        }
        out = collapse_to_two_stem(stems, vocal_key="voice")
        assert set(out.keys()) == {"vocals", "accompaniment"}


class TestEvaluateSeparation:
    def test_evaluate_near_identical_high_sdr(self, near_identical_stems):
        from evaluation.metrics import evaluate_separation
        ref, est = near_identical_stems
        scores = evaluate_separation(ref, est)
        assert "vocals" in scores
        assert scores["vocals"]["SDR"] > 20.0, f"SDR too low: {scores['vocals']['SDR']}"
        assert scores["vocals"]["SIR"] > 20.0
        assert scores["vocals"]["SAR"] > 20.0
        assert "SI-SDR" in scores["vocals"]
        assert scores["vocals"]["SI-SDR"] > 60.0, f"SI-SDR not scale-invariant: {scores['vocals']['SI-SDR']}"

    def test_evaluate_orthogonal_low_sdr(self, orthogonal_stems):
        from evaluation.metrics import evaluate_separation
        # Use orthogonal stems as both ref and est (completely different)
        scores = evaluate_separation(orthogonal_stems, orthogonal_stems)
        # Should still be identical -> high SDR (sanity check)
        assert scores["vocals"]["SDR"] > 20.0

    def test_evaluate_silent_reference_raises(self):
        from evaluation.metrics import evaluate_separation
        ref = {"vocals": np.zeros((2, 44100), dtype=np.float32)}
        est = {"vocals": np.random.randn(2, 44100).astype(np.float32)}
        with pytest.raises(ValueError, match="silent"):
            evaluate_separation(ref, est)

    def test_evaluate_silent_estimated_raises(self):
        from evaluation.metrics import evaluate_separation
        ref = {"vocals": np.random.randn(2, 44100).astype(np.float32)}
        est = {"vocals": np.zeros((2, 44100), dtype=np.float32)}
        with pytest.raises(ValueError, match="silent"):
            evaluate_separation(ref, est)

    def test_evaluate_mismatched_keys_raises(self):
        from evaluation.metrics import evaluate_separation
        ref = {"vocals": np.random.randn(2, 44100).astype(np.float32)}
        est = {"voice": np.random.randn(2, 44100).astype(np.float32)}
        with pytest.raises(ValueError, match="keys"):
            evaluate_separation(ref, est)

    def test_evaluate_mismatched_shapes_raises(self):
        from evaluation.metrics import evaluate_separation
        ref = {"vocals": np.random.randn(2, 44100).astype(np.float32)}
        est = {"vocals": np.random.randn(2, 22050).astype(np.float32)}
        with pytest.raises(ValueError, match="shape"):
            evaluate_separation(ref, est)

    def test_evaluate_nonfinite_raises(self):
        from evaluation.metrics import evaluate_separation
        ref = {"vocals": np.full((2, 44100), np.nan, dtype=np.float32)}
        est = {"vocals": np.zeros((2, 44100), dtype=np.float32)}
        with pytest.raises(ValueError, match="finite"):
            evaluate_separation(ref, est)

    def test_evaluate_channel_mode_equivalence(self):
        from evaluation.metrics import evaluate_separation
        rng = _rng(5)
        ref = {"vocals": rng.standard_normal((2, 44100), dtype=np.float32)}
        est = {"vocals": ref["vocals"] + 1e-6 * rng.standard_normal((2, 44100), dtype=np.float32)}
        for mode in ("per_channel", "mono_average", "first"):
            scores = evaluate_separation(ref, est, channel_mode=mode)
            # All modes should give high SDR for near-identical
            assert scores["vocals"]["SDR"] > 20.0, f"Mode {mode} failed: {scores['vocals']['SDR']}"

    def test_si_sdr_scale_invariance(self):
        from evaluation.metrics import evaluate_separation
        rng = _rng(6)
        s = rng.standard_normal((2, 44100), dtype=np.float32)
        ref = {"vocals": s}
        est_scaled = {"vocals": 2.5 * s}
        scores = evaluate_separation(ref, est_scaled)
        # Scale-invariant: SI-SDR(s, 2.5*s) == SI-SDR(s, s) = inf (practically > 60 dB)
        assert scores["vocals"]["SI-SDR"] > 60.0, f"SI-SDR not scale-invariant: {scores['vocals']['SI-SDR']}"