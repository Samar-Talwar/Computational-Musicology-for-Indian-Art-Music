# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Acoustic source separation pipeline for Indian classical music — benchmarking against the Sanidha baseline (arXiv:2501.06959). Four pipeline stages: spectrogram profiling & HPSS → neural model fine-tuning → waveform resynthesis → instrument classification. Benchmark target: SDR ~7.87 (Sanidha paper's fine-tuned Spleeter, 2-stem vocal/accompaniment split) — this is a locked comparative benchmark, not open-ended exploration. No 4-stem SDR may be compared to it without an explicit methodological note.

## Environment

- Python 3.13, Windows, venv at `./.venv/`
- Activate: `.\.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Key libraries: librosa, scipy, numpy, scikit-learn, matplotlib, mir_eval, mirdata, soundfile, h5py, pandas, jupyter, torch, demucs-infer

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

- Evaluation metrics: use `mir_eval.separation.bss_eval_sources` for SDR/SIR/SAR; compute zero-mean scale-invariant SI-SDR locally in `evaluation/metrics.py` since `mir_eval 0.8.2` omits SI-SDR and `bss_eval_sources` is deprecated (will be removed in 0.9).
- Audio I/O: `librosa.load()` for reading, `soundfile.write()` for writing
- Model checkpoints are gitignored (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`)
- Formatter: `black`

## Testing Philosophy

- **Strict TDD** for `preprocessing/`, `models/`, `evaluation/` — clear I/O contracts, write the failing test first
- **Exploratory-first** for spectrogram/HPSS parameter tuning in `notebooks/` — visualize and listen before writing tests; extract to `preprocessing/` with tests only once a technique is chosen
- **Synthetic test fixtures must match production dtype.** Any test signal generated via numpy (`np.sin`, `np.linspace`, etc.) defaults to `float64`, but real audio loaded via `librosa.load()` is `float32`. A test that passes on a `float64` synthetic fixture can silently fail once real `float32` audio flows through the same code path. Always explicitly cast synthetic fixtures to `float32` unless there's a documented reason not to.

## Git Workflow

- Feature branches per milestone: `milestone1/stft-hpss`, `milestone2/finetune`, etc.
- Use git worktrees for isolation on non-trivial features

## Mandatory Adversarial Plan Review

Before any implementation plan is presented for my approval, the plan must undergo an adversarial review.

The review must check:

1. **UNVERIFIED API CLAIMS**  
   For every imported function, class, method, attribute, or configuration value used by the plan:
   - verify that it actually exists with that name and location
   - if verification evidence is unavailable, mark it UNVERIFIED
   - never present an unverified API claim as established fact

2. **INTERNAL CONTRADICTIONS**  
   Trace every important variable, object, file, parameter, model output, and interface across every code block and step that uses it.
   Flag any mismatch in names, shapes, types, paths, signatures, or expected values.

3. **EXPECTATION IS NOT EVIDENCE**  
   Statements such as "Expected: PASS" are predictions, not evidence.
   Mark them as NOT YET RUN unless there is actual execution output proving the result.

4. **SILENT DEFAULTS AND HEURISTICS**  
   Flag behavior that depends on unstated assumptions, inferred array layouts, silent defaults, guessed parameters, or undocumented heuristics.
   Require explicit configuration or explicit failure where appropriate.

5. **SEQUENCING**  
   Check whether claimed verification actually happened before the code or plan step being labeled verified.
   Do not treat a proposed test as evidence.
   Do not treat planned exploration as completed exploration.

The reviewer must be adversarial rather than supportive.
Its job is to find flaws, contradictions, unsupported claims, and assumptions.
It must not defend the plan or rewrite it during review.

The review report must quote the exact plan text responsible for every finding.

A plan is considered reviewed only when:
- the reviewer reports no blocking issues, OR
- every blocking issue is fixed and the revised plan is reviewed again.

## Resolved Decisions
- **Spleeter → demucs-infer:** Replaced by `demucs-infer==4.2.2` (HDemucs, PyTorch), because Spleeter's TensorFlow is incompatible with Python 3.13.

## Open Decisions
- **Spleeter baseline reproduction method:** (a) reproduce Spleeter in an isolated old-Python venv/Docker purely to generate our own comparable SDR numbers under our own eval pipeline, or (b) cite Sanidha paper's published SDR (~7.87 target) directly, with an explicitly documented caveat that this compares our pipeline's evaluation against their pipeline's evaluation rather than an apples-to-apples reproduction. Not resolved yet — do not silently pick one.
- **2-stem vs 4-stem benchmark scope:** The ~7.87 SDR target is a 2-stem (vocal vs accompaniment) number. Our demucs-infer pipeline produces 4 stems (drums, bass, other, vocals). Decision needed: compare on a 2-stem basis (sum non-vocal stems into accompaniment) or define a separate 4-stem evaluation with its own baseline.
- **Artist-overlap leakage across Sanidha concerts:** All 5 concerts are local (61.5 GB). Current split: concerts 1, 2, 4, 5 for training / concert 3 for validation / no held-out test set. Audit whether artists overlap across concerts (unverified); decide test split.
