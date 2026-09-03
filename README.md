# Computational Musicology for Indian Art Music

**Acoustic Sound Isolation Pipeline optimized for Indian Classical Instrumentation**

Course project benchmarking a source-separation pipeline against the published
Spleeter/Sanidha baseline (arXiv:2501.06959, ISMIR 2024, Georgia Tech).

## Project Overview

Standard Western-trained audio source separation models fail on Indian classical
music — they misclassify tonal percussion (tabla, mridangam) as generic drums and
cannot separate overlapping microtonal glides from instruments like sitar, veena,
and sarod. This project builds and evaluates a separation + classification pipeline
specifically adapted for this domain.

## Pipeline Stages

1. **Spectrogram Profiling & HPSS Baseline** — DSP preprocessing, harmonic-percussive
   source separation as a classical baseline
2. **Neural Model Fine-Tuning** — domain adaptation of a separation network on the
   Sanidha dataset
3. **Waveform Resynthesis** — ISTFT/vocoder reconstruction from predicted masks
4. **Instrument Classification** — MFCC-based CNN classifier validating separation quality

## Datasets

- **Sanidha** — studio-quality Carnatic multi-track dataset (zero-bleed)
- **Saraga Multitrack Dataset** — real-world concert recordings (via `mirdata`)
- **IMIDataset** — used for the downstream classifier stage only

## Repository Structure

\`\`\`
data/raw/          # Original datasets (gitignored — not committed)
data/processed/    # Generated spectrograms/features (gitignored)
preprocessing/     # STFT, HPSS, normalization scripts
models/            # Neural network architecture + training code
notebooks/         # Exploration/visualization notebooks
experiments/       # Experiment configs and logs
evaluation/        # SDR/SI-SDR/SIR/SAR metrics, classifier evaluation
utils/             # Shared helper functions
docs/              # Research notes, architecture decisions, literature review
\`\`\`

## Setup

_(to be filled in once environment/dependencies are finalized)_

