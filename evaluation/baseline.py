"""Baseline evaluation — zero-training HDemucs 2-stem on Concert 3.

Design (4 fixes incorporated):
- Window bounds via soundfile.info + min_segment_sec 1.0.
- Silent-ref + silent-est skip; never crash evaluate_separation.
- Per-song mean/median + overall mean/median; None if 0 windows.
- JSON serializes with native Python types (float/str), mkdir parents.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import soundfile as sf

from models.separator import separate
from utils.sanidha import load_song, SongRecord
from evaluation.metrics import collapse_to_two_stem, evaluate_separation


def _energy(x: np.ndarray) -> float:
    return float(np.dot(x.ravel(), x.ravel()))


def _to_dict(stems_np: np.ndarray) -> dict[str, np.ndarray]:
    """Convert separate() output (4, 2, N) float32 -> dict."""
    # Order from demucs: drums(0), bass(1), other(2), vocals(3)
    return {
        "drums": stems_np[0].astype(np.float32),
        "bass": stems_np[1].astype(np.float32),
        "other": stems_np[2].astype(np.float32),
        "vocals": stems_np[3].astype(np.float32),
    }


def _is_silent(arr: np.ndarray, thresh: float = 1e-12) -> bool:
    return _energy(arr) <= thresh


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def evaluate_song_two_stem(
    record: SongRecord,
    *,
    segment_duration_sec: float = 30.0,
    max_segments: int | None = None,
    include_tanpura: bool = True,
    separator_fn: Callable[[np.ndarray, int], dict[str, np.ndarray] | np.ndarray] = separate,
    min_segment_sec: float = 1.0,
) -> dict[str, Any]:
    if segment_duration_sec <= 0:
        raise ValueError("segment_duration_sec must be > 0")

    # Determine total duration from first stem header (fix 1)
    first_path = next(iter(record.stem_files.values()))[0]
    info = sf.info(first_path)
    total_samples = info.frames
    total_sr = info.samplerate
    total_duration = total_samples / total_sr

    windows: list[dict[str, Any]] = []
    skipped_silent_ref_windows = 0
    skipped_silent_est_windows = 0

    i = 0
    while True:
        if max_segments is not None and i >= max_segments:
            break
        start_sec = i * segment_duration_sec
        if start_sec >= total_duration:
            break
        end_sec = start_sec + segment_duration_sec
        # Stop cleanly if trailing partial window shorter than min
        if (end_sec > total_duration) and (total_duration - start_sec < min_segment_sec):
            break

        # Load mix and ref stems for this window
        mix, ref_stems = load_song(
            record, include_tanpura=include_tanpura,
            start_sec=start_sec, duration_sec=min(segment_duration_sec, total_duration - start_sec),
            target_sr=total_sr,
        )

        # Collapse to 2-stem
        ref_2 = collapse_to_two_stem(ref_stems, vocal_key="vocals")

        # Silence handling fix 2: skip silent reference windows
        silent_ref = False
        for k in ("vocals", "accompaniment"):
            if _is_silent(ref_2[k]):
                silent_ref = True
                break
        if silent_ref:
            skipped_silent_ref_windows += 1
            i += 1
            continue

        # Call separator. separate() returns np.ndarray; wrap if needed.
        sep_out = separator_fn(mix, total_sr)
        if isinstance(sep_out, np.ndarray) and sep_out.ndim == 3:
            est_dict = _to_dict(sep_out)
        elif isinstance(sep_out, dict):
            est_dict = sep_out
        else:
            raise ValueError(f"separator_fn returned unexpected type: {type(sep_out)}")

        est_2 = collapse_to_two_stem(est_dict, vocal_key="vocals")

        # Silence handling fix 2 continued: skip silent estimated on non-silent ref
        silent_est = False
        for k in ("vocals", "accompaniment"):
            if _is_silent(est_2[k]):
                silent_est = True
                break
        if silent_est:
            skipped_silent_est_windows += 1
            i += 1
            continue

        # Evaluate
        scores = evaluate_separation(ref_2, est_2, channel_mode="per_channel")

        # Per-window metrics (mean across stems for quick view; store per-stem)
        window_result = {
            "window_idx": i,
            "start_sec": float(start_sec),
            "end_sec": float(min(end_sec, total_duration)),
            "skipped_silent_ref": False,
            "skipped_silent_est": False,
            "scores": scores,
        }
        windows.append(window_result)
        i += 1

    # Aggregate fix 3: per-song mean + median; None if 0 windows
    def _metrics_for(key: str):
        if not windows:
            return None
        vals = {stem: [] for stem in ("vocals", "accompaniment")}
        for w in windows:
            for stem in ("vocals", "accompaniment"):
                for metric in ("SDR", "SIR", "SAR", "SI-SDR"):
                    vals[stem].append(w["scores"][stem][metric])
        # Actually compute per-stem, per-metric mean/median
        result: dict[str, Any] = {}
        for stem in ("vocals", "accompaniment"):
            stem_result: dict[str, Any] = {}
            for metric in ("SDR", "SIR", "SAR", "SI-SDR"):
                arr = np.array([w["scores"][stem][metric] for w in windows], dtype=np.float32)
                stem_result[f"{metric}_mean"] = float(np.mean(arr))
                stem_result[f"{metric}_median"] = float(np.median(arr))
            result[stem] = stem_result
        return result

    per_song = _metrics_for("per_song") if windows else None

    return {
        "record_id": f"{record.concert_id}/{record.song_id}",
        "segment_duration_sec": float(segment_duration_sec),
        "max_segments": max_segments,
        "total_duration_sec": float(total_duration),
        "valid_windows": len(windows),
        "skipped_silent_ref_windows": skipped_silent_ref_windows,
        "skipped_silent_est_windows": skipped_silent_est_windows,
        "per_song_mean_median": per_song,
        "windows": windows,
    }


def run_concert_baseline(
    val_records: list[SongRecord],
    *,
    segment_duration_sec: float = 30.0,
    max_segments_per_song: int | None = None,
    include_tanpura: bool = True,
    output_json: Path | None = None,
    separator_fn: Callable[[np.ndarray, int], dict[str, np.ndarray] | np.ndarray] = separate,
) -> dict[str, Any]:
    per_song_results = []
    all_scores: list[dict] = []

    for rec in val_records:
        res = evaluate_song_two_stem(
            rec,
            segment_duration_sec=segment_duration_sec,
            max_segments=max_segments_per_song,
            include_tanpura=include_tanpura,
            separator_fn=separator_fn,
        )
        per_song_results.append(res)
        if res["valid_windows"] > 0 and res["per_song_mean_median"] is not None:
            all_scores.append(res["per_song_mean_median"])

    # Overall fix 3: pool all valid windows across songs -> global mean/median
    def _compute_overall_scores():
        all_window_scores = []
        for res in per_song_results:
            if res.get("valid_windows", 0) > 0 and res.get("windows"):
                all_window_scores.extend(res["windows"])
        if not all_window_scores:
            return None
        result_overall: dict[str, Any] = {}
        for stem in ("vocals", "accompaniment"):
            stem_result: dict[str, Any] = {}
            for metric in ("SDR", "SIR", "SAR", "SI-SDR"):
                vals = np.array([w["scores"][stem][metric] for w in all_window_scores], dtype=np.float32)
                stem_result[f"{metric}_mean"] = float(np.mean(vals))
                stem_result[f"{metric}_median"] = float(np.median(vals))
            result_overall[stem] = stem_result
        return result_overall

    overall = _compute_overall_scores()
    result = {
        "concert": "Concert03 (val)",
        "songs_evaluated": len(val_records),
        "per_song_results": per_song_results,
        "overall_mean_median": overall,
    }

    # Serialize fix 4
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(_serialize(result), f, indent=2)

    return result
