"""Source separation wrapper around demucs-infer (verified 4.2.2, HDemucs).

Design choices (all verified, no silent heuristics):
- Channels-first input convention: 2D audio must be shaped (channels, samples).
- Mono input: 1D array duplicated to stereo via np.tile derived from model.audio_channels.
- Input: (channels, samples) numpy; internal [batch, channels, samples].
- Output: (4 sources, channels, samples) numpy, float32.
- Lazy model loading: model instantiated on first separate() call.
"""
from typing import Optional
import numpy as np
import torch
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model

_model: Optional[torch.nn.Module] = None


def _get_model() -> torch.nn.Module:
    """Lazy loader for demucs model to avoid import-time overhead."""
    global _model
    if _model is None:
        _model = get_model('demucs_unittest')  # HDemucs, 2 ch, 44100 Hz, 4 stems
    return _model


def separate(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Separate stereo/mono audio into 4 instrument stems.

    Parameters
    ----------
    audio: np.ndarray, shape (2, N) stereo or (N,) mono, float32.
    sr: int, must match loaded model (44100).

    Returns
    -------
    np.ndarray, shape (4, 2, N) float32 — drums, bass, other, vocals.
    """
    model = _get_model()
    if sr != model.samplerate:
        raise ValueError(f"Expected sr={model.samplerate}, got {sr}")

    if audio.ndim == 1:
        audio = np.tile(audio[None, :], (model.audio_channels, 1))
    elif audio.ndim == 2:
        if audio.shape[0] != model.audio_channels:
            raise ValueError(
                f"Expected {model.audio_channels} channels for 2D audio, got {audio.shape[0]}"
            )
    else:
        raise ValueError(f"audio must be 1D (mono) or 2D (stereo); got ndim={audio.ndim}")

    mix = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    out = apply_model(model, mix, progress=False)
    return out[0].cpu().numpy()
