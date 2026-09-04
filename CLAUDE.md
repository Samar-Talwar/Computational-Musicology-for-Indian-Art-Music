# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Acoustic source separation pipeline for Indian classical music — benchmarking against the Spleeter/Sanidha baseline (arXiv:2501.06959). Four pipeline stages: spectrogram profiling & HPSS → neural model fine-tuning → waveform resynthesis → instrument classification. Benchmark target: SDR ~7.87 (Sanidha paper's fine-tuned Spleeter) — this is a locked comparative benchmark, not open-ended exploration.

## Environment

- Python 3.13, Windows, venv at `./venv/`
- Activate: `.\venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Key libraries: librosa, scipy, numpy, scikit-learn, matplotlib, mir_eval, mirdata, soundfile, h5py, pandas, jupyter

## Datasets

- **Sanidha** — studio-quality Carnatic multi-track (zero-bleed), primary training set
- **Saraga Multitrack** — concert recordings via `mirdata`
- **IMIDataset** — downstream classifier only
- **compIAM** — melody/rhythm/structure analysis and dataset loading only, NOT for source separation
- Audio goes in `data/raw/` (gitignored), features in `data/processed/` (gitignored)

## Repository Layout

```
preprocessing/   # STFT, HPSS, normalization
models/          # Network architecture + training
evaluation/      # SDR/SI-SDR/SIR/SAR metrics, classifier eval
utils/           # Shared helpers
notebooks/       # Exploration & visualization
experiments/     # Configs and logs
docs/            # Research notes, literature review
```

## Conventions

- Evaluation metrics: use `mir_eval` for SDR, SIR, SAR, SI-SDR (don't reimplement)
- Audio I/O: `librosa.load()` for reading, `soundfile.write()` for writing
- Model checkpoints are gitignored (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`)
- Formatter: `black`

## Testing Philosophy

- **Strict TDD** for `preprocessing/`, `models/`, `evaluation/` — clear I/O contracts, write the failing test first
- **Exploratory-first** for spectrogram/HPSS parameter tuning in `notebooks/` — visualize and listen before writing tests; extract to `preprocessing/` with tests only once a technique is chosen

## Git Workflow

- Feature branches per milestone: `milestone1/stft-hpss`, `milestone2/finetune`, etc.
- Use git worktrees for isolation on non-trivial features
