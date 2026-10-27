"""Evaluation metrics for source separation.

Conventions (all explicit, no silent heuristics):
- Input arrays: (channels, samples), float32. Stereo = 2 channels.
- mir_eval.separation.bss_eval_sources expects (nsrc, nsampl) mono.
  We support per-channel evaluation (default) and explicit mono collapse.
- SI-SDR computed locally (zero-mean scale-invariant), since mir_eval 0.8.2 omits it.
- bss_eval_sources is deprecated in mir_eval 0.8; using it with explicit warning.
"""

import warnings
from typing import Literal
import numpy as np
import mir_eval.separation


# Suppress the deprecation warning for bss_eval_sources - we use it deliberately
warnings.filterwarnings("ignore", category=FutureWarning, module="mir_eval.separation")


ChannelMode = Literal["per_channel", "mono_average", "first"]


def _validate_stems(
    stems: dict[str, np.ndarray],
    *,
    require_vocal_key: str | None = None,
    allow_empty: bool = False,
) -> tuple[int, int]:
    """Validate stem dictionary contract.

    Returns (n_channels, n_samples) if valid.
    """
    if not stems:
        raise ValueError("stems dictionary is empty")

    if require_vocal_key is not None and require_vocal_key not in stems:
        raise ValueError(f"vocal_key '{require_vocal_key}' not found in stems")

    # Check all arrays have same shape and dtype
    first = next(iter(stems.values()))
    if first.ndim != 2:
        raise ValueError(f"stem arrays must be 2D (channels, samples); got ndim={first.ndim}")
    n_channels, n_samples = first.shape

    for name, arr in stems.items():
        if arr.shape != (n_channels, n_samples):
            raise ValueError(f"stem '{name}' shape {arr.shape} != expected ({n_channels}, {n_samples})")
        if arr.dtype != np.float32:
            raise ValueError(f"stem '{name}' dtype {arr.dtype} != float32")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"stem '{name}' contains non-finite values (NaN/Inf)")

    if not allow_empty and len(stems) == 0:
        raise ValueError("stems dictionary is empty")

    return n_channels, n_samples


def _energy(x: np.ndarray) -> float:
    """Total energy of array (sum of squares)."""
    return float(np.dot(x.ravel(), x.ravel()))


def _is_silent(x: np.ndarray, threshold: float = 1e-12) -> bool:
    """Check if array is all-zeros / near-zero energy."""
    return _energy(x) <= threshold


def collapse_to_two_stem(
    stems: dict[str, np.ndarray],
    vocal_key: str = "vocals",
) -> dict[str, np.ndarray]:
    """Collapse an N-stem dictionary into 2-stem: {vocals, accompaniment}.

    Parameters
    ----------
    stems : dict[str, np.ndarray]
        Mapping stem_name -> array of shape (channels, samples), float32.
        Typical keys: HDemucs {"drums", "bass", "other", "vocals"},
        Sanidha {"vocals", "violin", "mridangam", "ghatam"} or 3-stem.
    vocal_key : str, default "vocals"
        Key to treat as the vocal stem.

    Returns
    -------
    dict[str, np.ndarray]
        {"vocals": (channels, samples), "accompaniment": (channels, samples)} float32.

    Raises
    ------
    ValueError
        If stems empty, vocal_key missing, shapes mismatch, dtype not float32,
        or no non-vocal stems to sum.
    """
    _validate_stems(stems, require_vocal_key=vocal_key)

    vocals = stems[vocal_key]
    non_vocal = {k: v for k, v in stems.items() if k != vocal_key}

    if not non_vocal:
        raise ValueError("no non-vocal stems to sum for accompaniment")

    accompaniment = sum(non_vocal.values())

    return {"vocals": vocals.copy(), "accompaniment": accompaniment.copy()}


def _si_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Zero-mean scale-invariant SDR (SI-SDR) for 1D or 2D arrays.

    Formula (per channel if 2D):
    1. Zero-mean both signals: s = s - mean(s), s_hat = s_hat - mean(s_hat)
    2. Projection: alpha = dot(s_hat, s) / dot(s, s)
    3. Target: s_target = alpha * s
    4. Noise: e_noise = s_hat - s_target
    5. SI-SDR = 10 * log10(dot(s_target, s_target) / dot(e_noise, e_noise))

    Parameters
    ----------
    reference : np.ndarray
        Shape (samples,) or (channels, samples)
    estimated : np.ndarray
        Same shape as reference

    Returns
    -------
    float
        SI-SDR in dB. Returns +inf if e_noise energy is 0 (perfect match).
    """
    if reference.ndim == 1:
        reference = reference[np.newaxis, :]
        estimated = estimated[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    if reference.shape != estimated.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {estimated.shape}")

    n_channels = reference.shape[0]
    si_sdr_values = []

    for ch in range(n_channels):
        s = reference[ch] - np.mean(reference[ch])
        s_hat = estimated[ch] - np.mean(estimated[ch])

        if _is_silent(s):
            raise ValueError("reference signal has zero/near-zero energy")
        if _is_silent(s_hat):
            raise ValueError("estimated signal has zero/near-zero energy")

        alpha = float(np.dot(s_hat, s) / np.dot(s, s))
        s_target = alpha * s
        e_noise = s_hat - s_target

        e_energy = _energy(e_noise)
        if e_energy == 0:
            return float("inf")

        s_target_energy = _energy(s_target)
        si_sdr_val = 10.0 * np.log10(s_target_energy / e_energy)
        si_sdr_values.append(si_sdr_val)

    return float(np.mean(si_sdr_values)) if not squeeze else si_sdr_values[0]


def _to_mono_per_channel(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert (channels, samples) stems to (samples,) per channel."""
    return {k: v for k, v in stems.items()}  # pass through for per-channel evaluation


def _collapse_to_mono(stems: dict[str, np.ndarray], mode: ChannelMode) -> dict[str, np.ndarray]:
    """Collapse stereo (2, N) to mono (N,) per stem."""
    out = {}
    for k, v in stems.items():
        if mode == "mono_average":
            out[k] = np.mean(v, axis=0)
        elif mode == "first":
            out[k] = v[0]
        elif mode == "sum":
            out[k] = np.sum(v, axis=0)
        else:
            raise ValueError(f"unknown channel_mode: {mode}")
    return out


def evaluate_separation(
    reference_stems: dict[str, np.ndarray],
    estimated_stems: dict[str, np.ndarray],
    *,
    compute_permutation: bool = False,
    channel_mode: ChannelMode = "per_channel",
) -> dict[str, dict[str, float]]:
    """Compute SDR, SIR, SAR (and SI-SDR) per stem.

    Parameters
    ----------
    reference_stems : dict[str, np.ndarray]
        Ground truth stems, each (channels, samples), float32.
    estimated_stems : dict[str, np.ndarray]
        Estimated stems, same keys, same shapes, float32.
    compute_permutation : bool, default False
        If True, let mir_eval find best permutation.
    channel_mode : {"per_channel", "mono_average", "first"}, default "per_channel"
        How to handle stereo (channels, samples) for bss_eval_sources:
        - "per_channel": evaluate each channel separately, average metric scores
        - "mono_average": average channels to mono before evaluation
        - "first": use first channel only

    Returns
    -------
    dict[str, dict[str, float]]
        {stem_name: {"SDR": ..., "SIR": ..., "SAR": ..., "SI-SDR": ...}}

    Raises
    ------
    ValueError
        Mismatched keys, shapes, non-finite values, silent reference/estimated stem.
    """
    # Validate both dictionaries
    ref_ch, ref_samp = _validate_stems(reference_stems)
    est_ch, est_samp = _validate_stems(estimated_stems)

    if set(reference_stems.keys()) != set(estimated_stems.keys()):
        raise ValueError(
            f"keys mismatch: ref={set(reference_stems.keys())} != est={set(estimated_stems.keys())}"
        )
    if ref_ch != est_ch or ref_samp != est_samp:
        raise ValueError(f"shape mismatch: ref=({ref_ch},{ref_samp}) est=({est_ch},{est_samp})")

    # Fixed stem order for consistent indexing
    stem_names = sorted(reference_stems.keys())
    n_src = len(stem_names)

    # Collect metrics per stem
    results: dict[str, dict[str, float]] = {}

    if channel_mode == "per_channel":
        # Evaluate each channel separately with bss_eval_sources
        all_ref_ch = []
        all_est_ch = []

        for ch in range(ref_ch):
            ref_ch_stems = np.stack([reference_stems[name][ch] for name in stem_names])
            est_ch_stems = np.stack([estimated_stems[name][ch] for name in stem_names])

            # Check silent stems per channel
            for i, name in enumerate(stem_names):
                if _is_silent(ref_ch_stems[i]):
                    raise ValueError(f"reference stem '{name}' channel {ch} is silent")
                if _is_silent(est_ch_stems[i]):
                    raise ValueError(f"estimated stem '{name}' channel {ch} is silent")

            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref_ch_stems, est_ch_stems, compute_permutation=compute_permutation
            )

            for i, name in enumerate(stem_names):
                if name not in results:
                    results[name] = {"SDR": 0.0, "SIR": 0.0, "SAR": 0.0, "SI-SDR": 0.0}
                results[name]["SDR"] += sdr[i]
                results[name]["SIR"] += sir[i]
                results[name]["SAR"] += sar[i]

        # Average across channels
        for name in results:
            results[name]["SDR"] /= ref_ch
            results[name]["SIR"] /= ref_ch
            results[name]["SAR"] /= ref_ch

        # SI-SDR: zero-mean scale-invariant, averaged across channels
        for name in stem_names:
            ref_ch_stems = np.stack([reference_stems[name][ch] for ch in range(ref_ch)])
            est_ch_stems = np.stack([estimated_stems[name][ch] for ch in range(ref_ch)])
            results[name]["SI-SDR"] = _si_sdr(ref_ch_stems, est_ch_stems)

    else:
        # Collapse to mono first
        ref_mono = _collapse_to_mono(reference_stems, channel_mode)
        est_mono = _collapse_to_mono(estimated_stems, channel_mode)

        ref_stack = np.stack([ref_mono[name] for name in stem_names])
        est_stack = np.stack([est_mono[name] for name in stem_names])

        # Check silent stems
        for i, name in enumerate(stem_names):
            if _is_silent(ref_stack[i]):
                raise ValueError(f"reference stem '{name}' is silent after {channel_mode}")
            if _is_silent(est_stack[i]):
                raise ValueError(f"estimated stem '{name}' is silent after {channel_mode}")

        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            ref_stack, est_stack, compute_permutation=compute_permutation
        )

        for i, name in enumerate(stem_names):
            results[name] = {
                "SDR": float(sdr[i]),
                "SIR": float(sir[i]),
                "SAR": float(sar[i]),
                "SI-SDR": _si_sdr(ref_stack[i], est_stack[i]),
            }

    return results