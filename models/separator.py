"""Source separation wrapper around demucs-infer (verified 4.2.2, HDemucs).

Design choices (all verified, no silent heuristics):
- Channels-first input convention: 2D audio must be shaped (channels, samples).
- Mono input: 1D array duplicated to stereo via np.tile derived from model.audio_channels.
- Input: (channels, samples) numpy; internal [batch, channels, samples].
- Output: (4 sources, channels, samples) numpy, float32.
- Lazy model loading: model instantiated on first separate() call.
"""
from __future__ import annotations
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


# =============================================================================
# HDemucs2Stem — 2-stem transfer learning module (Carnatic fine-tuned)
# =============================================================================

class HDemucs2Stem(torch.nn.Module):
    """HTDemucs adapted for 2-stem (vocals, accompaniment) fine-tuning.

    Wraps the pretrained HDemucs backbone (4 sources) and projects to 2 stems
    via acoustic summation: accompaniment = drums + bass + other.

    Parameters
    ----------
    freeze_encoders: bool, default True
        If True, freeze all parameters with "encoder" or "tencoder" in their name.
    checkpoint_path: Optional[str | Path], default None
        Path to a .pt checkpoint containing {'model_state_dict': ...} or bare state dict.
        If provided, loads weights into the backbone before any freezing.
    """
    def __init__(
        self,
        freeze_encoders: bool = True,
        checkpoint_path: Optional[str | Path] = None,
    ):
        super().__init__()

        # Load pretrained BagOfModels and extract single model backbone
        bag = get_model('htdemucs')
        if hasattr(bag, 'models') and len(bag.models) > 0:
            self.backbone = bag.models[0]
        else:
            self.backbone = bag

        # Load checkpoint if provided (state dict must match backbone keys)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing:
                print(f"Checkpoint missing keys: {len(missing)} (e.g., {missing[:3]})")
            if unexpected:
                print(f"Checkpoint unexpected keys: {len(unexpected)} (e.g., {unexpected[:3]})")

        # Freeze encoder / tencoder parameters
        if freeze_encoders:
            for name, param in self.backbone.named_parameters():
                if "encoder" in name or "tencoder" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2-stem separation.

        Parameters
        ----------
        mix: torch.Tensor, shape (B, 2, T) float32 — batch of stereo mixtures.

        Returns
        -------
        torch.Tensor, shape (B, 2, 2, T) float32 —
            dim 1: [vocals, accompaniment], each (2, T) stereo.
        """
        # backbone forward returns (B, 4, 2, T) — [drums, bass, other, vocals]
        out = self.backbone(mix)

        # Acoustic projection to 2 stems
        vocals = out[:, 3:4, :, :]          # (B, 1, 2, T)
        accompaniment = out[:, 0:3, :, :].sum(dim=1, keepdim=True)  # (B, 1, 2, T)

        # Stack to (B, 2, 2, T) — [vocals, accompaniment]
        return torch.cat([vocals, accompaniment], dim=1)