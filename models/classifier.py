"""TimbreCNN classifier for Indian classical instrument recognition.

Architecture verified on Colab (Milestone 4).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import librosa


INSTRUMENT_CLASSES = [
    "bass",
    "drums",
    "flute",
    "guitar",
    "piano",
    "sarod",
    "sitar",
    "tabla",
    "veena",
    "violin",
]


class TimbreCNN(nn.Module):
    """CNN for timbre-based instrument classification.

    Expects input shaped (B, 1, 40, T) or (B, 40, T) — MFCC features.
    Returns logits shaped (B, num_classes).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Conv block 1: 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global pooling + classifier head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Shape (B, 1, 40, T) or (B, 40, T).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # Accept (B, 40, T) and add channel dim
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    hop_length: int = 512,
    target_samples: int = 44100,
) -> np.ndarray:
    """Extract MFCC features from audio, resampled/padded to 2.0 seconds.

    Parameters
    ----------
    audio: np.ndarray
        Input audio signal (1D), float32.
    sr: int
        Sample rate of input audio (default 22050).
    n_mfcc: int
        Number of MFCC coefficients (default 40).
    hop_length: int
        Hop length for STFT (default 512).
    target_samples: int
        Target duration in samples at sr (default 44100 = 2.0s at 22050 Hz).

    Returns
    -------
    np.ndarray
        MFCC features of shape (40, 87), dtype float32.
    """
    # Resample to target sample rate if needed
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Pad or trim to target_samples (2.0s at 22050 Hz = 44100 samples)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )

    # Expected shape: (40, 87) for 44100 samples at hop_length=512
    # (44100 / 512) + 1 = 87 frames
    assert mfcc.shape == (n_mfcc, 87), f"Expected ({n_mfcc}, 87), got {mfcc.shape}"
    return mfcc.astype(np.float32)