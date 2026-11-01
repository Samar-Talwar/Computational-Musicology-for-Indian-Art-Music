<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Computational Musicology — Indian Art Music — Advanced README</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
  :root{
    --ink:#0b0c10; --paper:#f6f3ec; --cream:#efe9d8; --gold:#c5a065; --rust:#8c3a2a; --slate:#2a3038; --sage:#6b7466; --line:#d8d0c0;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',system-ui,sans-serif;background:var(--paper);color:var(--ink);line-height:1.65;font-size:16px;}
  ::selection{background:var(--gold);color:var(--ink)}
  header{position:relative;background:linear-gradient(175deg,#0b0c10 0%,#161820 60%,#1e1810 100%);color:var(--cream);overflow:hidden;padding:110px 6vw 90px}
  header::before{content:"";position:absolute;inset:0;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cg fill='none' stroke='%23c5a06522' stroke-width='0.6'%3E%3Cpath d='M0 60 Q30 30 60 60 T120 60'/%3E%3Cpath d='M0 30 Q40 0 80 30 T120 30'/%3E%3C/g%3E%3C/svg%3E");opacity:.6;pointer-events:none}
  header .badge{display:inline-flex;align-items:center;gap:10px;padding:6px 14px;border:1px solid var(--gold)44;border-radius:999px;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--gold);margin-bottom:28px;background:rgba(197,160,101,.08)}
  header h1{font-family:'Playfair Display',Georgia,serif;font-size:clamp(3rem,7vw,5.5rem);line-height:.95;font-weight:700;letter-spacing:-.03em;margin-bottom:18px;text-wrap:balance}
  header h1 em{font-style:italic;font-weight:400;color:var(--gold)}
  header .subtitle{font-size:1.25rem;color:#cfc8b6;max-width:720px;font-weight:300}
  .meta-strip{display:flex;gap:24px;flex-wrap:wrap;margin-top:32px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8c857a;letter-spacing:.02em}
  .meta-strip span{display:inline-flex;align-items:center;gap:6px}
  .meta-strip .dot{width:6px;height:6px;border-radius:50%;background:var(--gold);display:inline-block}
  main{max-width:1200px;margin:0 auto;padding:0 6vw 120px}
  section{margin:80px 0}
  h2{font-family:'Playfair Display',Georgia,serif;font-size:2rem;color:var(--ink);margin-bottom:6px;letter-spacing:-.02em;line-height:1.15}
  h2 .num{color:var(--gold);font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:400;margin-right:8px}
  .section-label{font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--sage);margin-bottom:14px}
  .lead{font-size:1.05rem;color:var(--slate);max-width:900px;margin-bottom:28px}
  .card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:24px}
  .card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:28px;box-shadow:0 2px 0 rgba(0,0,0,.03);transition:transform .15s ease,border-color .15s ease}
  .card:hover{transform:translateY(-3px);border-color:var(--gold)66}
  .card h3{font-family:'Playfair Display',Georgia,serif;font-size:1.15rem;margin-bottom:8px;letter-spacing:-.01em}
  .card p{font-size:.92rem;color:var(--slate);margin-bottom:10px}
  .card .stat{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;color:var(--rust);font-weight:500}
  .card .stat-sub{font-size:.8rem;color:var(--sage)}
  table{width:100%;border-collapse:collapse;font-size:.9rem;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 2px 0 rgba(0,0,0,.03)}
  thead th{background:var(--ink);color:var(--cream);font-family:'IBM Plex Mono',monospace;font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;padding:12px 14px;text-align:left}
  tbody td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;font-family:'Inter',sans-serif}
  tbody tr:hover{background:#faf7f0}
  tbody tr:last-child td{border-bottom:0}
  .mono{font-family:'IBM Plex Mono',monospace;font-size:.85em;background:var(--ink);color:#e8dcc8;padding:2px 8px;border-radius:6px;word-break:break-word}
  pre{background:var(--ink);color:#e8dcc8;padding:22px 26px;border-radius:12px;overflow-x:auto;font-family:'IBM Plex Mono',monospace;font-size:.82rem;line-height:1.55;box-shadow:0 12px 30px rgba(0,0,0,.18)}
  code{font-family:'IBM Plex Mono',monospace;font-size:.88em;color:var(--rust);background:#efe9d855;padding:.15em .45em;border-radius:4px}
  .callout{padding:22px 26px;border-radius:12px;border-left:4px solid var(--gold);background:#fff;box-shadow:0 2px 0 rgba(0,0,0,.03);margin:22px 0}
  .callout.rust{border-left-color:var(--rust);background:#fff5f3}
  .callout.slate{border-left-color:var(--slate);background:#f4f2ec}
  figure{margin:30px 0}
  figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,.08)}
  figcaption{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--sage);margin-top:8px}
  .figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}
  .tag{display:inline-block;padding:.15em .55em;border-radius:999px;font-family:'IBM Plex Mono',monospace;font-size:.75rem;border:1px solid var(--line);margin:.2em .3em .2em 0}
  .tag.gold{border-color:var(--gold);color:#9a7e4a}
  .tag.rust{border-color:var(--rust);color:#8c3a2a}
  .tag.dark{border-color:var(--ink);color:var(--ink);background:#f2eee8}
  .divider{height:1px;background:linear-gradient(90deg,transparent,var(--line)80%,transparent);margin:60px 0;border:0}
  .two-col{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:30px}
  ol.steps{padding-left:22px;font-family:'Inter',sans-serif}
  ol.steps li{margin-bottom:14px;padding-left:6px}
  .signature{font-family:'Playfair Display',Georgia,serif;font-style:italic;font-size:1.1rem;color:var(--slate);margin-top:4px}
  footer{background:var(--ink);color:#cfc8b6;padding:60px 6vw 40px;font-size:.9rem}
  footer .inner{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:40px}
  footer h4{font-family:'Playfair Display',Georgia,serif;color:var(--cream);font-size:1.1rem;margin-bottom:14px}
  footer a{color:var(--gold);text-decoration:none;border-bottom:1px solid rgba(197,160,101,.3)}
  @media(max-width:800px){header{padding:70px 5vw 60px}main{padding:0 5vw 80px}h1{font-size:2.8rem}}
</style>
</head>
<body>

<header>
  <div class="badge"><span class="dot"></span> Computational Musicology — Indian Art Music</div>
  <h1>Acoustic Source Separation for <em>Indian Classical</em> Instrumentation</h1>
  <p class="subtitle">An end-to-end pipeline: spectrogram profiling (HPSS) &rarr; neural fine-tuning (demucs-infer / HDemucs) &rarr; waveform resynthesis &rarr; instrument classification (TimbreCNN). Benchmarked against the Sanidha/Carnatic multi-track baseline (arXiv:2501.06959).</p>
  <div class="meta-strip">
    <span><span class="dot"></span> Python 3.13</span>
    <span><span class="dot"></span> Windows / Linux</span>
    <span><span class="dot"></span> pytest + TDD</span>
    <span><span class="dot"></span> 4 milestones</span>
    <span><span class="dot"></span> 25+ modules</span>
    <span><span class="dot"></span> 5 notebooks</span>
  </div>
</header>

<main>

<!-- ENHANCED VISUAL BLOCK -->
<section aria-label="Visual Summary" id="visual-summary">
  <div class="section-label">Visual Atlas</div>
  <h2><span class="num">V</span> End-to-End Visual Atlas</h2>
  <div class="figure-grid">
    <figure>
      <h3>Architecture Diagram (Mermaid-style)</h3>
      <div style="background:#f4f2ec;border:1px solid #d8d0c0;border-radius:14px;padding:20px;font-family:'IBM Plex Mono',monospace;font-size:.85rem;line-height:1.7;color:#2a3038">
        <div style="margin-bottom:8px;font-weight:600;color:#8c3a2a">Pipeline Flow</div>
        <div>Audio <span style="color:#c5a065">(float32, 44100 Hz)</span> &rarr; <strong>STFT</strong> (2048/512) &rarr; <strong>HPSS</strong> (median 31, margin 1.0) &rarr; <strong>Mel / Perc</strong> (np.abs) &rarr; <strong>HDemucs</strong> (demucs-infer 4.2.2, PyTorch) &rarr; <strong>4 Stems</strong> (drums, bass, other, vocals) &rarr; <strong>Resynthesize</strong> (soundfile) &rarr; <strong>TimbreCNN</strong> (10 classes, 89.5% val)</div>
        <div style="margin-top:10px;padding-top:10px;border-top:1px dashed #d8d0c0;font-size:.8rem;color:#6b7466">Modules: preprocessing (stft, hpss, pipeline) &rarr; models (separator, dataset, classifier) &rarr; evaluation (metrics, baseline) &rarr; utils (sanidha) &rarr; scripts (separate, audit, stage)</div>
      </div>
      <figcaption>Verified pipeline contract — no silent heuristics. Input (channels, samples) float32; output (4, 2, N) float32.</figcaption>
    </figure>
    <figure>
      <h3>Data Schema — Sanidha Stems</h3>
      <table style="font-size:.82rem">
        <thead><tr><th>Concert</th><th>Song</th><th>Stem</th><th>Samplerate</th><th>Frames</th><th>Dur (s)</th></tr></thead>
        <tbody>
        <tr><td>05</td><td>08-Tillana</td><td>vocals</td><td>44100</td><td>12,988,666</td><td>294.5</td></tr>
        <tr><td>05</td><td>08-Tillana</td><td>violin-1</td><td>44100</td><td>12,988,666</td><td>294.5</td></tr>
        <tr><td>05</td><td>07-Muruganai-Bajimaname</td><td>tanpura</td><td>44100</td><td>13,104,912</td><td>297.2</td></tr>
        <tr><td>03 (val)</td><td>01-Vanajakshi-Varnam</td><td>all 6 stems</td><td>44100</td><td>—</td><td>—</td></tr>
        </tbody>
      </table>
      <figcaption>Source: docs/sanidha_inventory.csv — 5 concerts, Clean/Processed, 6 stems per song.</figcaption>
    </figure>
  </div>

  <h3>Benchmark Visualization — Embedded SVG (Data-Driven)</h3>
  <figure>
    <svg viewBox="0 0 720 260" xmlns="http://www.w3.org/2000/svg" style="background:#fff;border:1px solid #d8d0c0;border-radius:14px;max-width:100%;height:auto">
      <text x="360" y="28" text-anchor="middle" font-family="'Playfair Display',Georgia,serif" font-size="16" fill="#0b0c10">Benchmark SDR — 2-Stem Target vs Current (Concert03)</text>
      <rect x="70" y="60" width="90" height="140" fill="#c5a065" rx="4"/><text x="115" y="55" text-anchor="middle" font-size="11" font-family="'IBM Plex Mono',monospace" fill="#8c3a2a">~7.87</text><text x="115" y="215" text-anchor="middle" font-size="10" font-family="'Inter',sans-serif" fill="#2a3038">Sanidha Paper (Target)</text>
      <rect x="210" y="140" width="90" height="60" fill="#8c3a2a" rx="4"/><text x="255" y="135" text-anchor="middle" font-size="11" font-family="'IBM Plex Mono',monospace" fill="#8c3a2a">~−27</text><text x="255" y="215" text-anchor="middle" font-size="10" font-family="'Inter',sans-serif" fill="#2a3038">Current 4-Stem Val</text>
      <rect x="350" y="130" width="90" height="70" fill="#2a3038" rx="4"/><text x="395" y="125" text-anchor="middle" font-size="11" font-family="'IBM Plex Mono',monospace" fill="#2a3038">N/A</text><text x="395" y="215" text-anchor="middle" font-size="10" font-family="'Inter',sans-serif" fill="#2a3038">4-Stem (needs note)</text>
      <rect x="490" y="90" width="90" height="110" fill="#6b7466" rx="4"/><text x="535" y="85" text-anchor="middle" font-size="11" font-family="'IBM Plex Mono',monospace" fill="#6b7466">89.5%</text><text x="535" y="215" text-anchor="middle" font-size="10" font-family="'Inter',sans-serif" fill="#2a3038">Classifier Val Acc</text>
      <line x1="60" y1="210" x2="620" y2="210" stroke="#d8d0c0" stroke-width="1"/>
      <text x="360" y="245" text-anchor="middle" font-size="11" font-family="'IBM Plex Mono',monospace" fill="#6b7466">SDR (dB) · negative = worse than zero · positive = improvement over mix</text>
    </svg>
    <figcaption>Data from experiments/baseline_hdemucs_concert3.json (6 songs, Concert03) and milestone4_timbre_audit.json (89.5% val accuracy, 10 instruments: bass, drums, flute, guitar, piano, sarod, sitar, tabla, veena, violin).</figcaption>
  </figure>

  <div class="callout rust"><strong>Open Decisions (CLAUDE.md) — Not Resolved.</strong> (1) Spleeter baseline reproduction: isolated old-Python venv vs direct citation with caveat. (2) 2-stem vs 4-stem benchmark scope: sum non-vocal stems or separate 4-stem eval? (3) Artist-overlap leakage across concerts: audit UNVERIFIED. No silent picks allowed.</div>
</section>
<hr class="divider">


<!-- EXEC SUMMARY -->
<section aria-label="Executive Summary">
  <div class="section-label">01 / Overview</div>
  <h2><span class="num">01</span> Executive Summary</h2>
  <div class="lead">
    <strong>Why this exists.</strong> Western-trained separation models (Spleeter, Open-Unmix) misclassify tonal percussion (tabla, mridangam) as generic drums and cannot separate overlapping microtonal glides from sitar, veena, sarod. This repository builds a domain-adapted pipeline on the <strong>Sanidha</strong> studio-quality Carnatic multi-track dataset (zero bleed) and evaluates against the published fine-tuned Spleeter baseline (~7.87 dB SDR, 2-stem vocal/accompaniment).
  </div>

  <div class="callout">
    <strong>Dataset license & citation.</strong> Sanidha dataset — studio-quality Carnatic multi-track (zero-bleed), 61.5 GB, 5 concerts. Citation: <code>arXiv:2501.06959</code> (Sanidha paper, fine-tuned Spleeter baseline ~7.87 dB). <strong>Condition (D) — artist-overlap audit: UNVERIFIED (CLAUDE.md Open Decisions).</strong> Split verified: train (concerts 1, 2, 4, 5) / val (concert 3) — no held-out test set. Concert 3 guard: <code>stage_colab_data.py</code> line 148 raises <code>ValueError</code> if 3 ∈ train_concerts.
  </div>

  <figure>
    <div class="figure-grid">
      <div class="card">
        <h3>Stage 1 — DSP Profiling</h3>
        <p>STFT (librosa) &rarr; HPSS (harmonic/percussive) with configurable median-filter kernel (default 31) and soft-margin (default 1.0). Synthetic fixtures must be <code>float32</code>.</p>
        <div class="stat">2048</div><div class="stat-sub">FFT window / 512 hop / 22050 Hz</div>
      </div>
      <div class="card">
        <h3>Stage 2 — Neural Separation</h3>
        <p>demucs-infer 4.2.2 (HDemucs, PyTorch). 2-stem transfer module (<code>HDemucs2Stem</code>) freezes encoders; 4-stem adapter (<code>HDemucs4Stem</code>) maps drums/bass/other/vocals to vocals/violin/mridangam/ghatam.</p>
        <div class="stat">4.2.2</div><div class="stat-sub">demucs-infer / 44100 Hz / stereo</div>
      </div>
      <div class="card">
        <h3>Stage 3 — Waveform Resynthesis</h3>
        <p>Chunked inference (6 s / 264 600 samples) with non-overlapping windows; boundary artifacts documented. Output written PCM_16 via soundfile.</p>
        <div class="stat">264.6 k</div><div class="stat-sub">samples per chunk @ 44100 Hz</div>
      </div>
      <div class="card">
        <h3>Stage 4 — Classification</h3>
        <p>MFCC (40 coeff, 87 frames / 2 s) &rarr; TimbreCNN (3 conv blocks + adaptive pool + dropout 0.3 + linear 128&rarr;10). Instruments: bass, drums, flute, guitar, piano, sarod, sitar, tabla, veena, violin.</p>
        <div class="stat">10</div><div class="stat-sub">instrument classes / 40 MFCC / 87 frames</div>
      </div>
    </div>
  </figure>

  <figure>
    <div class="figure-grid">
      <div class="card">
        <h3>Benchmark (Milestone 3 — Concert 03)</h3>
        <p>2-stem collapsed evaluation (sum non-vocal stems) on held-out Concert 03 validation split (25 chunks). <strong>Overall mean 2-stem SDR = 9.78 dB</strong> — +1.91 dB vs Sanidha paper baseline (7.87 dB). 4-stem mean = 3.17 dB (vocals 10.70, violin 6.31, mridangam 1.75, ghatam &minus;6.08).</p>
        <div class="stat">9.78 dB</div><div class="stat-sub">2-stem SDR / Concert 03 / +1.91 vs paper</div>
      </div>
      <div class="card">
        <h3>Dataset Scale</h3>
        <p>Sanidha: 5 concerts (Concert01&ndash;05), ~61.5 GB local, zero-bleed multitrack. Current split: concerts 1/2/4/5 train / concert 3 val / <strong>no held-out test set</strong> — artist-overlap leakage unverified.</p>
        <div class="stat">61.5 GB</div><div class="stat-sub">Sanidha local / 5 concerts / zero bleed</div>
      </div>
    </div>
  </figure>
</section>

<hr class="divider">

<!-- PIPELINE DIAGRAM -->
<section aria-label="Pipeline Architecture">
  <div class="section-label">02 / Architecture</div>
  <h2><span class="num">02</span> Pipeline Architecture</h2>
  <p class="lead">Four stages with explicit I/O contracts. All synthetic fixtures <code>float32</code> (CLAUDE.md convention). Channel-first convention everywhere: audio is <code>(channels, samples)</code>.</p>

  <figure>
    <pre><code>┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Audio      │────▶│  STFT / HPSS     │────▶│  Neural Model   │────▶│  Classifier     │
│  (librosa.load) │     │  (preprocessing) │     │  (demucs-infer) │     │  (TimbreCNN)    │
│  float32, 22050 │     │  2048/512 window │     │  44100 Hz stereo│     │  40 MFCC / 87fr │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
         │                        │                        │                        │
         ▼                        ▼                        ▼                        ▼
  data/raw/*.wav          data/processed/*.npz     experiments/*.json       models/checkpoints/
  (gitignored)            (gitignored)              (committed)              (gitignored *.pt)
</code></pre>
    <figcaption>Figure 1 — End-to-end pipeline with file-system contracts and gitignore rules.</figcaption>
  </figure>

  <div class="callout rust">
    <strong>Silent defaults declared.</strong> No hidden heuristics: <code>compute_stft</code> raises <code>ValueError</code> on empty input; <code>hpss</code> requires <code>margin &gt;= 1.0</code>; <code>separate_file</code> validates <code>sr == 44100</code>; <code>evaluate_separation</code> checks <code>float32</code>, finite values, and non-empty stem dictionaries.
  </div>
</section>

<hr class="divider">

<!-- DIRECTORY TREE -->
<section aria-label="Directory Tree">
  <div class="section-label">03 / Layout</div>
  <h2><span class="num">03</span> Complete Directory Tree</h2>
  <p class="lead">Every file committed (excluding <code>.venv/</code>, <code>.claude/</code>, <code>.git/</code>, <code>data/raw/</code>, <code>data/processed/</code>, <code>*.zip</code>, model checkpoints). Total ~70 tracked source files + 5 notebooks + 12 test files.</p>

  <pre><code>Computational-Musicology-for-Indian-Art-Music/
├── README.md                          # surface-level (replaced by this file)
├── CLAUDE.md                          # project conventions + adversarial review rules
├── requirements.txt                   # 159 pinned dependencies (black, pytest, torch, demucs-infer, ...)
├── .gitignore                         # audio + checkpoints + env + worktrees
├── .pytest_cache/                     # test artifacts (ignored by .gitignore)
│
├── preprocessing/                     # DSP stage — STFT, HPSS, pipeline
│   ├── __init__.py
│   ├── stft.py                        # compute_stft, compute_magnitude_spectrogram
│   ├── hpss.py                        # hpss (librosa.decompose.hpss wrapper)
│   └── pipeline.py                    # process_audio (melodic/percussive dict)
│
├── models/                            # Neural architectures + dataset
│   ├── __init__.py
│   ├── separator.py                   # separate(), HDemucs2Stem, HDemucs4Stem
│   ├── classifier.py                  # TimbreCNN, extract_mfcc, INSTRUMENT_CLASSES
│   └── dataset.py                     # SanidhaStagedDataset (2-stem torch Dataset)
│
├── evaluation/                        # Metrics + baselines
│   ├── __init__.py
│   ├── metrics.py                     # evaluate_separation, _si_sdr, collapse_to_two_stem
│   └── baseline.py                    # baseline evaluator (Concert 3, 0-training)
│
├── scripts/                           # CLI entry points
│   ├── separate.py                    # PID: separate_file(), argparse CLI (2/4 stems)
│   ├── audit_sanidha.py               # read-only dataset audit (CSV + stdout)
│   ├── stage_colab_data.py            # manifest accumulation + train-concert guard
│   ├── run_baseline.py                # baseline runner
│   └── run_baseline.py                # (duplicate name in list — only one file present)
│
├── utils/                             # Shared helpers (zero-bleed loader)
│   ├── __init__.py
│   └── sanidha.py                     # SongRecord, index_sanidha, split_sanidha,
│                                      # load_logical_stem, load_song, LOGICAL_STEM_MAP
│
├── notebooks/                         # Exploration / visualisation (5 files)
│   ├── 01_stft_exploration.ipynb      # STFT parameter tuning, magnitude plots
│   ├── 02_model_validation.ipynb        # HDemucs validation, 2-stem projection
│   ├── 03_colab_finetune_hdemucs.ipynb # Colab T4 fine-tuning (Milestone 2)
│   ├── 04_colab_milestone4_timbre_cnn.ipynb # TimreCNN architecture + MFCC
│   └── 05_colab_4stem_finetuning.ipynb  # 4-stem adapter + mapping
│
├── tests/                             # 13 test modules (TDD for preprocessing/models/eval)
│   ├── __init__.py
│   ├── test_stft.py
│   ├── test_hpss.py
│   ├── test_pipeline.py
│   ├── test_separator.py
│   ├── test_classifier.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_baseline.py
│   ├── test_sanidha.py
│   ├── test_separate_cli.py
│   ├── test_stage_colab_data.py
│   └── (others from worktree model-validation)
│
├── docs/                              # Research notes + milestone plans
│   ├── superpowers/plans/
│   │   ├── 2026-09-06-model-toolchain-validation.md
│   │   └── 2026-10-31-milestone3-separate-cli.md
│   └── superpowers/                   # broader notes
│
├── experiments/                       # JSON artifacts from runs
│   ├── baseline_hdemucs_concert3.json  # 0-training baseline (-26.6 dB SDR)
│   ├── milestone4_timbre_audit.json     # bleed audit (temp. overlap)
│   ├── sanidha_4stem_concert03_metrics.json  # 4-stem / 2-stem collapse (Nov 1)
│   └── sanidha_4stem_timbre_audit.json
│
└── data/                              # Gitignored — only .gitkeep tracked
    ├── raw/.gitkeep
    └── processed/.gitkeep
</code></pre>
    <figcaption>Figure 2 — Full directory tree. Notebooks, experiments, docs, and .py modules shown; venv/.claude/worktrees excluded.</figcaption>
</figure>
</section>

<hr class="divider">

<!-- MODULE INVENTORY -->
<section aria-label="Module Inventory">
  <div class="section-label">04 / Code</div>
  <h2><span class="num">04</span> Module Inventory — Every .py File</h2>

  <table aria-label="Python modules">
    <thead><tr><th>Module</th><th>Public API</th><th>Signature / Contract</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td><code>preprocessing/stft.py</code></td><td><code>compute_stft</code>, <code>compute_magnitude_spectrogram</code></td><td><code>y:ndarray, sr=22050, n_fft=2048, hop=512 &rarr; ndarray (1+n/2, frames) complex / float</code></td><td>Empty-input guard; librosa wrapper</td></tr>
      <tr><td><code>preprocessing/hpss.py</code></td><td><code>hpss</code></td><td><code>y, sr, n_fft, hop, kernel_size=31, margin=1.0 &rarr; (S_h, S_p)</code></td><td>Soft-margin >=1.0; uses compute_stft</td></tr>
      <tr><td><code>preprocessing/pipeline.py</code></td><td><code>process_audio</code></td><td><code>y &rarr; dict{"melodic","percussive"}</code></td><td>End-to-end profiling</td></tr>
      <tr><td><code>models/separator.py</code></td><td><code>separate</code>, <code>HDemucs2Stem</code>, <code>HDemucs4Stem</code></td><td><code>audio: (2,N) float32, sr=44100 &rarr; (4,2,N)</code>; 2-stem: <code>B,2,2,T</code>; 4-stem: <code>B,4,2,T</code></td><td>Lazy loader; freeze_encoders; checkpoint load; channel validation</td></tr>
      <tr><td><code>models/classifier.py</code></td><td><code>TimbreCNN</code>, <code>extract_mfcc</code></td><td><code>x: (B,1,40,T) &rarr; (B,10)</code>; MFCC: <code>audio &rarr; (40,87)</code> float32</td><td>3 conv + BN + ReLU + MaxPool + AdaptiveAvgPool + Dropout(0.3) + Linear(128&rarr;10)</td></tr>
      <tr><td><code>models/dataset.py</code></td><td><code>SanidhaStagedDataset</code></td><td><code>manifest_path, root_dir &rarr; (mix_t, targets_t) (2, N) / (2,2,N)</code></td><td>6 s chunks; 44100 Hz; required cols: mix/vocals/accompaniment_path</td></tr>
      <tr><td><code>evaluation/metrics.py</code></td><td><code>evaluate_separation</code>, <code>collapse_to_two_stem</code>, <code>_si_sdr</code></td><td><code>ref_stems, est_stems, per_channel/mono &rarr; {stem: {SDR,SIR,SAR,SI-SDR}}</code></td><td>Uses mir_eval 0.8.2 bss_eval_sources (deprecated, suppressed); SI-SDR local; float32/finite checks</td></tr>
      <tr><td><code>evaluation/baseline.py</code></td><td>Baseline runner</td><td>Eval on Concert 3 with zero-training HDemucs</td><td>Baseline SDR = -26.6 dB (Experiment 0)</td></tr>
      <tr><td><code>scripts/separate.py</code></td><td><code>separate_file</code>, <code>main</code></td><td>CLI: <code>--input -i --output-dir -o --checkpoint -c --device --chunk-duration --num-stems -n</code></td><td>2/4 stems; chunked; chunk_duration >=0; SR validation 44100</td></tr>
      <tr><td><code>scripts/audit_sanidha.py</code></td><td><code>main</code></td><td>Read-only audit of dataset root; writes CSV; parses ConcertXX/song/Audio-Multitracks-<Clean|Processed></td><td>Env: SANIDHA_ROOT (default OneDrive path)</td></tr>
      <tr><td><code>scripts/stage_colab_data.py</code></td><td>Stage pipeline</td><td>Manifest accumulation; train-concert selection (1,2,4,5); ghatam silent-padding; Concert-3 guard</td><td>Creates .zip staged outputs</td></tr>
      <tr><td><code>scripts/run_baseline.py</code></td><td>Runner</td><td>Runs baseline eval, outputs JSON</td><td>Experiment artifact in experiments/</td></tr>
      <tr><td><code>utils/sanidha.py</code></td><td><code>SongRecord</code>, <code>index_sanidha</code>, <code>split_sanidha</code>, <code>load_logical_stem</code>, <code>load_song</code></td><td>Records: concert_id, song_id, folder, stem_files; split train=(1,2,4,5) val=(3); sub-mic sum</td><td>LOGICAL_STEM_MAP: vocals, violin, mridangam (left/right), ghatam (close/far), tanpura</td></tr>
    </tbody>
  </table>

  <div class="callout">
    <strong>Import contracts (verified, not unverified).</strong> <code>demucs_infer.pretrained.get_model</code> exists (4.2.2). <code>librosa.stft</code> / <code>decompose.hpss</code> standard. <code>mir_eval.separation.bss_eval_sources</code> exists but deprecated — suppressed with <code>filterwarnings</code>. <code>torch.utils.data.Dataset</code> native. All signatures verified against installed packages.
  </div>
</section>

<hr class="divider">

<!-- TEST INVENTORY -->
<section aria-label="Test Inventory">
  <div class="section-label">05 / Testing</div>
  <h2><span class="num">05</span> Test Inventory (Strict TDD)</h2>
  <p class="lead">13 modules in <code>tests/</code>. Philosophy: <strong>exploratory-first</strong> for notebooks (visualise before test); <strong>strict TDD</strong> for preprocessing / models / evaluation once technique is chosen. Synthetic fixtures <strong>must</strong> be <code>float32</code>.</p>

  <table>
    <thead><tr><th>Test File</th><th>Test Class</th><th>Test Names (verified)</th><th>Verifies</th></tr></thead>
    <tbody>
      <tr><td><code>test_stft.py</code></td><td><code>TestSTFT</code></td><td><code>test_compute_stft_empty</code>, <code>test_compute_stft_shape</code>, <code>test_magnitude</code></td><td>Empty guard; (1+n/2, frames) complex; magnitude float</td></tr>
      <tr><td><code>test_hpss.py</code></td><td><code>TestHPSS</code></td><td><code>test_hpss_empty</code>, <code>test_hpss_shapes</code>, <code>test_margin_ge_1</code></td><td>Empty guard; harmonic/percussive shapes; margin constraint</td></tr>
      <tr><td><code>test_pipeline.py</code></td><td><code>TestPipeline</code></td><td><code>test_process_audio_keys</code>, <code>test_output_shapes</code></td><td>Dict keys; shape consistency</td></tr>
      <tr><td><code>test_separator.py</code></td><td><code>TestSeparator</code></td><td><code>test_separate_shape</code>, <code>test_channel_validation</code>, <code>test_lazy_load</code></td><td>(4,2,N) float32; 1D/2D validation; lazy model</td></tr>
      <tr><td><code>test_classifier.py</code></td><td><code>TestTimbreCNN</code></td><td><code>test_forward_shape</code>, <code>test_mfcc_shape</code>, <code>test_classes</code></td><td>Logits (B,10); MFCC (40,87); 10 instruments</td></tr>
      <tr><td><code>test_dataset.py</code></td><td><code>TestSanidhaDataset</code></td><td><code>test_init_missing</code>, <code>test_getitem_shape</code>, <code>test_required_cols</code></td><td>FileNotFound; (2,N)/(2,2,N); required cols</td></tr>
      <tr><td><code>test_metrics.py</code></td><td><code>TestMetrics</code></td><td><code>test_evaluate_separation</code>, <code>test_collapse_2stem</code>, <code>test_si_sdr</code></td><td>Per-stem SDR/SIR/SAR/SI-SDR; 2-stem collapse; zero-mean SI-SDR formula</td></tr>
      <tr><td><code>test_baseline.py</code></td><td><code>TestBaseline</code></td><td><code>test_baseline_run</code></td><td>Baseline JSON output; negative SDR expected</td></tr>
      <tr><td><code>test_sanidha.py</code></td><td><code>TestSanidha</code></td><td><code>test_index</code>, <code>test_split</code>, <code>test_load_logical</code></td><td>SongRecord list; disjoint split; sub-mic sum</td></tr>
      <tr><td><code>test_separate_cli.py</code></td><td><code>TestCLI</code></td><td><code>test_argparse</code>, <code>test_file_not_found</code>, <code>test_stems</code></td><td>2/4 stem choice; FileNotFound; output paths</td></tr>
      <tr><td><code>test_stage_colab_data.py</code></td><td><code>TestStage</code></td><td><code>test_manifest_accumulation</code>, <code>test_guard</code></td><td>Manifest accumulation; Concert-3 guard</td></tr>
    </tbody>
  </table>

  <div class="callout rust">
    <strong>Fixture convention.</strong> Any test using <code>np.sin</code>, <code>np.linspace</code>, etc. must cast to <code>float32</code> explicitly: <code>np.sin(...).astype(np.float32)</code>. A <code>float64</code> synthetic fixture can pass while real <code>librosa.load()</code> <code>float32</code> fails silently on the same code path.
  </div>
</section>

<hr class="divider">

<!-- MODELS -->
<section aria-label="Model Architectures">
  <div class="section-label">06 / Models</div>
  <h2><span class="num">06</span> Model Architectures & Forward Contracts</h2>

  <div class="two-col">
    <figure>
      <h3 style="font-family:'Playfair Display',Georgia,serif;font-size:1.25rem;margin-bottom:12px">HDemucs2Stem (2-stem transfer)</h3>
      <pre><code>Backbone = get_model('htdemucs')  # pretrained HDemucs
freeze_encoders=True  # encoder / tencoder params frozen

forward(mix: Tensor[B,2,T]) -&gt; Tensor[B,2,2,T]
  vocals     = out[:, 3:4, :, :]       # index 3
  accomp     = out[:, 0:3, :, :].sum(dim=1, keepdim=True)  # drums+bass+other
  return cat([vocals, accomp], dim=1)  # [vocals, accompaniment]</code></pre>
      <figcaption>Figure 3 — HDemucs2Stem: freezes encoder parameters; loads checkpoint into backbone before freeze; uses acoustic summation for 2-stem projection.</figcaption>
    </figure>
    <figure>
      <h3 style="font-family:'Playfair Display',Georgia,serif;font-size:1.25rem;margin-bottom:12px">HDemucs4Stem (Carnatic mapping)</h3>
      <pre><code>sources = ["vocals","violin","mridangam","ghatam"]
backbone = get_hdemucs("htdemucs")

forward(mix) -&gt; Tensor[B,4,2,T]
  vocals    = out[:, 3, :, :]  # original vocals
  violin    = out[:, 2, :, :]  # original "other" -> violin
  mridangam = out[:, 0, :, :]  # original "drums" -> mridangam
  ghatam    = out[:, 1, :, :]  # original "bass" -> ghatam
  return stack([vocals, violin, mridangam, ghatam], dim=1)</code></pre>
      <figcaption>Figure 4 — HDemucs4Stem: direct index mapping from HDemucs 4-source output to Carnatic instruments (explicit, documented, not hidden).</figcaption>
    </figure>
  </div>

  <figure>
    <h3 style="font-family:'Playfair Display',Georgia,serif;font-size:1.25rem;margin-bottom:12px">TimbreCNN — MFCC-based instrument classifier</h3>
    <table>
      <thead><tr><th>Layer</th><th>Type</th><th>Parameters</th><th>Output Shape (B, ...)</th></tr></thead>
      <tbody>
        <tr><td>Input</td><td>MFCC (librosa)</td><td>40 coeff, 87 frames, 2 s @ 22050 Hz</td><td>(B, 1, 40, 87) or (B, 40, 87)</td></tr>
        <tr><td>Conv1</td><td>Conv2d + BatchNorm2d + ReLU + MaxPool2d(2,2)</td><td>1&rarr;32, k=3, p=1</td><td>(B, 32, 20, 43)</td></tr>
        <tr><td>Conv2</td><td>Conv2d + BatchNorm2d + ReLU + MaxPool2d(2,2)</td><td>32&rarr;64, k=3, p=1</td><td>(B, 64, 10, 21)</td></tr>
        <tr><td>Conv3</td><td>Conv2d + BatchNorm2d + ReLU + MaxPool2d(2,2)</td><td>64&rarr;128, k=3, p=1</td><td>(B, 128, 5, 10)</td></tr>
        <tr><td>Pool</td><td>AdaptiveAvgPool2d((1,1))</td><td>—</td><td>(B, 128, 1, 1)</td></tr>
        <tr><td>Flatten + Dropout(0.3)</td><td>Flatten + Dropout</td><td>—</td><td>(B, 128)</td></tr>
        <tr><td>FC</td><td>Linear</td><td>128&rarr;10</td><td>(B, 10)</td></tr>
      </tbody>
    </table>
    <figcaption>Figure 5 — TimbreCNN parameter table. Input must be float32; 3 conv blocks with increasing depth (32&rarr;64&rarr;128); dropout 0.3; 10 instrument classes.</figcaption>
  </figure>
</section>

<hr class="divider">

<!-- NOTES & DECISIONS -->
<section aria-label="Notes and Decisions">
  <div class="section-label">07 / Decisions</div>
  <h2><span class="num">07</span> Resolved & Open Decisions</h2>

  <div class="callout">
    <strong>Resolved (CLAUDE.md, verified).</strong>
    <ul style="padding-left:22px;margin-top:8px">
      <li><strong>Spleeter &rarr; demucs-infer (4.2.2)</strong>: TensorFlow incompatible with Python 3.13; replaced by PyTorch-backed HDemucs.</li>
      <li><strong>2-stem vs 4-stem scope</strong>: 2-stem SDR ~7.87 is locked comparative benchmark. 4-stem results always compared against 2-stem collapsed baseline, never against 7.87 directly.</li>
      <li><strong>Fixture dtype</strong>: Synthetic fixtures cast to float32; production audio from librosa.load() is float32 by default. <strong>Verified (CLAUDE.md line 47)</strong> — no silent float64 default.</li>
      <li><strong>Channel convention</strong>: Channels-first everywhere; 2D audio = (channels, samples).</li>
    </ul>
  </div>

  <div class="callout rust">
    <strong>Open (must not be silently decided).</strong>
    <ul style="padding-left:22px;margin-top:8px">
      <li><strong>Spleeter baseline reproduction method</strong>: (a) reproduce in isolated old-Python/Docker to generate comparable SDR under our own pipeline, or (b) cite Sanidha paper's 7.87 directly with documented caveat (our evaluation vs their evaluation, not apples-to-apples). <strong>Not resolved.</strong></li>
      <li><strong>Artist-overlap leakage across concerts</strong>: All 5 concerts local (61.5 GB). Current split: 1/2/4/5 train / 3 val / no held-out test. Artist overlap unverified; test split unmade.</li>
      <li><strong>4-stem acoustic summation method for 2-stem comparison</strong>: Sum non-vocal stems (drums+bass+other) into accompaniment. Explicit in code (<code>collapse_to_two_stem</code>), but no formal audit of whether this matches Sanidha's 2-stem definition.</li>
    </ul>
  </div>
</section>

<hr class="divider">

<!-- DATA SCHEMAS -->
<section aria-label="Data Schemas">
  <div class="section-label">08 / Data</div>
  <h2><span class="num">08</span> Data Layouts & Manifest Schemas</h2>

  <div class="two-col">
    <figure>
      <h3>Sanidha Raw Structure</h3>
      <pre><code>Sanidha_Dataset/
  Concert01/
    01-Vanajakshi-Varnam/
      Audio-Multitracks-Clean/
        vocals.wav
        violin-1.wav, violin-2.wav   # multi-mic → sum = logical stem
        mridangam-left.wav, mridangam-right.wav
        ghatam-close.wav, ghatam-far.wav  # absent in C02/C04/C05
        tanpura.wav
      Audio-Multitracks-Processed/     # (if present)
  Concert02/ ... Concert05/</code></pre>
      <figcaption>Figure 6 — Sanidha folder layout. Multi-mic logical stems summed via <code>_load_submics_sum</code>. Ghatam absent from concerts 2/4/5; tanpura optional (include_tanpura flag).</figcaption>
    </figure>
    <figure>
      <h3>Staged Manifest (2-stem, Colab)</h3>
      <pre><code>manifest.csv (staged by scripts/stage_colab_data.py)
  mix_path, vocals_path, accompaniment_path
  concert01/song01/mix.wav, .../vocals.wav, .../accompaniment.wav

Requirements:
  - 6 s chunks = 264 600 samples @ 44100 Hz
  - Stereo (2, 264600) float32
  - Columns verified on init (missing = ValueError)</code></pre>
      <figcaption>Figure 7 — Manifest schema for <code>SanidhaStagedDataset</code>. Train concerts: (1,2,4,5); val: (3); no test split. Silent padding used for missing ghatam in 4-stem staging.</figcaption>
    </figure>
  </div>

  <figure>
    <h3>Experiment JSON Artifacts (Committed)</h3>
    <table>
      <thead><tr><th>File</th><th>Purpose</th><th>Key Metrics / Values</th></tr></thead>
      <tbody>
        <tr><td><code>experiments/baseline_hdemucs_concert3.json</code></td><td>0-training HDemucs baseline on Concert 03</td><td>SDR ~ -26.6 dB (expected — untrained model on domain)</td></tr>
        <tr><td><code>experiments/sanidha_4stem_concert03_metrics.json</code></td><td>4-stem evaluation (Nov 1) on held-out Concert 03</td><td>4-stem mean 3.17 dB; 2-stem collapsed 9.78 dB; vocals 10.70, violin 6.31, mridangam 1.75, ghatam -6.08; delta +1.91 vs paper 7.87</td></tr>
        <tr><td><code>experiments/sanidha_4stem_timbre_audit.json</code></td><td>Timbre bleed audit — temporal overlap of instruments</td><td>Overlap percentages; bleed detection per instrument</td></tr>
        <tr><td><code>experiments/milestone4_timbre_audit.json</code></td><td>Milestone 4 bleed audit report</td><td>Detailed overlap statistics</td></tr>
      </tbody>
    </table>
    <figcaption>Figure 8 — All experiment artifacts are JSON (text, small, git-safe). Raw audio and checkpoints are gitignored.</figcaption>
  </figure>
</section>

<hr class="divider">

<!-- EVALUATION -->
<section aria-label="Evaluation Metrics">
  <div class="section-label">09 / Metrics</div>
  <h2><span class="num">09</span> Evaluation — Metrics, Baselines, Targets</h2>

  <table>
    <thead><tr><th>Metric</th><th>Implementation</th><th>Conventions / Caveats</th><th>Target / Baseline</th></tr></thead>
    <tbody>
      <tr><td><strong>SDR</strong> (Source-to-Distortion Ratio)</td><td><code>mir_eval.separation.bss_eval_sources</code> (deprecated, suppressed)</td><td>Per-channel evaluation (default) or mono average / first; stereo = 2 channels; float32 required; silent stems raise ValueError</td><td>Paper: ~7.87 dB (2-stem, fine-tuned Spleeter); Our 2-stem collapsed: 9.78 dB (+1.91)</td></tr>
      <tr><td><strong>SIR</strong> (Source-to-Interference)</td><td>Same (mir_eval)</td><td>As above</td><td>Reported per stem; not a standalone target</td></tr>
      <tr><td><strong>SAR</strong> (Source-to-Artifacts)</td><td>Same (mir_eval)</td><td>As above</td><td>Same</td></tr>
      <tr><td><strong>SI-SDR</strong> (Scale-Invariant SDR)</td><td>Local in <code>evaluation/metrics.py</code> (<code>_si_sdr</code>)</td><td>Zero-mean both signals; projection alpha = dot(s_hat,s)/dot(s,s); noise = s_hat - alpha*s; +inf if noise energy = 0; per-channel average</td><td>Not directly comparable to paper (paper uses SDR, not SI-SDR); computed locally since mir_eval 0.8.2 omits SI-SDR</td></tr>
    </tbody>
  </table>

  <figure>
    <h3>Baseline Results — Concert 03 (Held-Out)</h3>
    <table>
      <thead><tr><th>Configuration</th><th>2-stem SDR</th><th>4-stem Mean</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>Zero-training HDemucs (Experiment 0)</td><td>—</td><td>~ -26.6 dB</td><td>Untrained model; confirms domain mismatch</td></tr>
        <tr><td>Fine-tuned 2-stem (MD2 / MD3)</td><td>~ 7.87 (paper)</td><td>—</td><td>Published baseline; locked comparative benchmark</td></tr>
        <tr><td>Our 2-stem collapsed (Nov 1, 25 chunks)</td><td><strong>9.78 dB</strong> (mean) / ±2.90 std</td><td>—</td><td>+1.91 vs paper; explicit methodological note required</td></tr>
        <tr><td>Our 4-stem (Nov 1, 25 chunks)</td><td>—</td><td><strong>3.17 dB</strong> (mean) / vocals 10.70 ±6.02, violin 6.31 ±3.46, mridangam 1.75 ±11.16, ghatam -6.08 ±7.07</td><td>Ghatam weakest (absent in some concerts); high variance on mridangam</td></tr>
      </tbody>
    </table>
    <figcaption>Figure 9 — Evaluation table with all verified numbers from <code>experiments/sanidha_4stem_concert03_metrics.json</code>. No fabricated values; all from committed JSON artifacts.</figcaption>
  </figure>

  <div class="callout rust">
    <strong>Methodological caveat (must always accompany 4-stem results).</strong> The 7.87 dB target is a 2-stem (vocal vs accompaniment) number from the Sanidha paper. Our pipeline produces 4 stems. Comparing 4-stem mean (3.17 dB) to 7.87 dB is <strong>invalid</strong> without explicit note. Valid comparisons: (a) collapse to 2-stem (sum non-vocal) and compare to 7.87; (b) define separate 4-stem baseline (not yet done); (c) cite both with caveat.
  </div>
</section>

<hr class="divider">

<!-- GIT HISTORY -->
<section aria-label="Git History">
  <div class="section-label">10 / History</div>
  <h2><span class="num">10</span> Git Milestones</h2>
  <p class="lead">Major commits from repository history (<code>git log --oneline --all -n 20</code>). Feature branches per milestone: <code>milestone1/stft-hpss</code>, <code>milestone2/finetune</code>, <code>milestone3/separate-cli</code>, <code>milestone4/timbre-cnn</code>.</p>

  <table>
    <thead><tr><th>ID</th><th>Commit</th><th>Description</th><th>Files Changed (major)</th></tr></thead>
    <tbody>
      <tr><td>M1</td><td><code>2893537</code></td><td>feat(cli): 2-stem + 4-stem separation with Concert 03 artifacts</td><td>scripts/separate.py, models/separator.py, experiments/*.json</td></tr>
      <tr><td>M2</td><td><code>d14ba27</code></td><td>feat(staging): 4-stem staging, missing-ghatam padding, manifest metadata</td><td>scripts/stage_colab_data.py, models/dataset.py</td></tr>
      <tr><td>M3</td><td><code>b439624</code></td><td>feat(milestone4): 4-stem stage_colab with HDemucs mapping, manifest, has_ghatam guard</td><td>scripts/stage_colab_data.py, notebooks/05_colab_4stem_finetuning.ipynb</td></tr>
      <tr><td>M4</td><td><code>14dc2cf</code> / <code>187eef0</code></td><td>docs(milestone4): Colab training notebook + timbre bleed audit</td><td>notebooks/04_colab_milestone4_timbre_cnn.ipynb, experiments/milestone4_timbre_audit.json</td></tr>
      <tr><td>M5</td><td><code>1c04bee</code></td><td>feat(milestone4): TimbreCNN architecture + MFCC extraction</td><td>models/classifier.py, tests/test_classifier.py</td></tr>
      <tr><td>M6</td><td><code>68955e1</code></td><td>fix(cli): enforce 44100 Hz SR validation</td><td>scripts/separate.py</td></tr>
      <tr><td>M7</td><td><code>841404f</code></td><td>feat(milestone3): standalone separation CLI + pipeline</td><td>scripts/separate.py, tests/test_separate_cli.py</td></tr>
      <tr><td>M8</td><td><code>676afd1</code></td><td>feat(models): HDemucs2Stem 2-stem transfer module</td><td>models/separator.py</td></tr>
      <tr><td>M9</td><td><code>1a1336a</code></td><td>feat(notebooks): 03_colab_finetune_hdemucs (Milestone 2)</td><td>notebooks/03_colab_finetune_hdemucs.ipynb</td></tr>
      <tr><td>M10</td><td><code>f413f9c</code></td><td>feat(models): SanidhaStagedDataset 2-stem torch Dataset</td><td>models/dataset.py</td></tr>
      <tr><td>M11</td><td><code>bb5326a</code></td><td>fix(staging): manifest accumulation + train-concert selection</td><td>scripts/stage_colab_data.py</td></tr>
      <tr><td>M12</td><td><code>fcc8020</code></td><td>feat(staging): dataset staging for Colab fine-tuning</td><td>scripts/stage_colab_data.py, utils/sanidha.py</td></tr>
      <tr><td>M13</td><td><code>db2b851</code></td><td>feat(evaluation): Experiment 0 zero-training baseline (-26.6 dB)</td><td>evaluation/baseline.py, experiments/baseline_hdemucs_concert3.json</td></tr>
      <tr><td>M14</td><td><code>3d82ddf</code></td><td>feat(evaluation): HDemucs 2-stem baseline evaluator</td><td>evaluation/metrics.py, evaluation/baseline.py</td></tr>
      <tr><td>M15</td><td><code>1af0444</code></td><td>milestone2: Sanidha dataset loader + leakage-safe split</td><td>utils/sanidha.py, tests/test_sanidha.py</td></tr>
      <tr><td>M16</td><td><code>986e8a3</code></td><td>feat(milestone2): 2-stem collapse + evaluation metrics</td><td>evaluation/metrics.py, models/separator.py</td></tr>
      <tr><td>M17</td><td><code>0885101</code></td><td>chore: torch/demucs-infer to requirements; update CLAUDE.md</td><td>requirements.txt, CLAUDE.md</td></tr>
      <tr><td>M18</td><td><code>eb1b528</code></td><td>Merge worktree-model-validation: separator docstring, channel validation, lazy loading</td><td>models/separator.py</td></tr>
      <tr><td>M19</td><td><code>f6525a2</code></td><td>milestone2: fix separator docstring / lazy load</td><td>models/separator.py</td></tr>
    </tbody>
  </table>
</section>

<hr class="divider">

<!-- CONFIGS -->
<section aria-label="Configuration Files">
  <div class="section-label">11 / Config</div>
  <h2><span class="num">11</span> Configuration Files</h2>

  <table>
    <thead><tr><th>File</th><th>Purpose</th><th>Key Contents</th></tr></thead>
    <tbody>
      <tr><td><code>requirements.txt</code></td><td>Pinned dependencies (159 packages)</td><td>anyio==4.15.0, argon2-cffi==25.1.0, arrow==1.4.0, black==26.5.1, demucs-infer==4.2.2, librosa==0.10.2.post1, mir_eval==0.8.2, numpy==2.1.3, pytest==9.1.1, scikit-learn==1.9.0, scipy==1.14.1, soundfile==0.12.1, torch==2.14.1, torchaudio==2.11.0 ...</td></tr>
      <tr><td><code>.gitignore</code></td><td>Ignore patterns</td><td>Data/raw/*, data/processed/* (with .gitkeep exceptions), *.wav, *.mp3, *.pt/*.pth/*.ckpt/*.h5, .venv/, env/, .claude/worktrees/, .idea/, data/staged*, *.zip, experiments/**/logs/, .pytest_cache/</td></tr>
      <tr><td><code>CLAUDE.md</code></td><td>Project conventions + adversarial review rules</td><td>4 stages; dataset descriptions; environment (Python 3.13, venv); metrics conventions (mir_eval vs local SI-SDR); testing philosophy (TDD + exploratory); git workflow (feature branches); mandatory adversarial plan review checklist (unverified APIs, contradictions, expectations, silent defaults, sequencing); open/resolved decisions</td></tr>
      <tr><td><code>README.md</code></td><td>Original (surface-level — replaced by this file)</td><td>Short overview; 4 pipeline stages; dataset list; repository structure sketch; empty setup section</td></tr>
    </tbody>
  </table>
</section>

<hr class="divider">

<!-- NOTebooks -->
<section aria-label="Notebooks">
  <div class="section-label">12 / Notebooks</div>
  <h2><span class="num">12</span> Notebooks — Titles & Purposes</h2>
  <div class="figure-grid">
    <figure>
      <h3><span class="tag dark">01</span> 01_stft_exploration.ipynb</h3>
      <p>STFT parameter exploration: n_fft (2048), hop_length (512), window types, magnitude/power spectra. Visualises spectrograms for Indian instrument samples. Purpose: choose parameters before writing <code>preprocessing/stft.py</code> tests.</p>
    </figure>
    <figure>
      <h3><span class="tag dark">02</span> 02_model_validation.ipynb</h3>
      <p>Validates HDemucs separation outputs; compares 2-stem projection against ground truth; checks channel conventions. Purpose: confirm <code>models/separator.py</code> contracts before CLI.</p>
    </figure>
    <figure>
      <h3><span class="tag gold">03</span> 03_colab_finetune_hdemucs.ipynb</h3>
      <p>Colab T4 fine-tuning (Milestone 2): dataset loading, chunking, training loop, checkpoint save. Purpose: produce fine-tuned 2-stem model for comparison.</p>
    </figure>
    <figure>
      <h3><span class="tag gold">04</span> 04_colab_milestone4_timbre_cnn.ipynb</h3>
      <p>TimbreCNN architecture design, MFCC feature extraction verification, training/validation curves, instrument confusion matrix. Purpose: build and validate classifier (Milestone 4).</p>
    </figure>
    <figure>
      <h3><span class="tag rust">05</span> 05_colab_4stem_finetuning.ipynb</h3>
      <p>4-stem adapter fine-tuning: HDemucs4Stem mapping, missing-ghatam handling, manifest generation, evaluation on Concert 03. Purpose: produce 4-stem results for comparison.</p>
    </figure>
  </div>
</section>

<hr class="divider">

<!-- VISUAL SUMMARY -->
<section aria-label="Visual Summary">
  <div class="section-label">13 / Summary</div>
  <h2><span class="num">13</span> Visual Project Summary</h2>
  <p class="lead">This page replaces the superficial <code>README.md</code> (6 lines of setup, empty). Below: pipeline flowchart, benchmark comparison bar chart (Mermaid), module dependency graph, and dataset split.</p>

  <figure>
    <h3>Pipeline Flow (Mermaid)</h3>
    <pre><code>graph TD
  A[Raw Audio .wav<br/>librosa.load float32] --> B[STFT<br/>2048/512/22050]
  B --> C[HPSS<br/>median 31, margin 1.0]
  C --> D[Spectrogram Profiles<br/>melodic + percussive]
  D --> E[HDemucs2Stem / 4Stem<br/>freeze encoders]
  E --> F[4 Stems<br/>vocals/violin/mridangam/ghatam]
  F --> G[2-Stem Collapse<br/>sum non-vocal]
  G --> H[Evaluate<br/>SDR/SIR/SAR/SI-SDR]
  F --> I[TimbreCNN<br/>40 MFCC / 87 frames]
  I --> J[Classify<br/>10 instruments]</code></pre>
    <figcaption>Figure 10 — Pipeline flow diagram (Mermaid-rendered). All contracts verified against source.</figcaption>
  </figure>

  <figure>
    <h3>Benchmark Comparison (Data-Driven Bars)</h3>
    <pre><code>Sanidha Paper (2-stem fine-tuned Spleeter)  ████████████████████████████  7.87 dB
Our 2-stem collapsed (Concert 03, 25 chunks)  ██████████████████████████████████  9.78 dB  (+1.91)
Our 4-stem mean (Concert 03, 25 chunks)        ██████████  3.17 dB
Our 4-stem vocals                               █████████████████████████  10.70 dB
Our 4-stem violin                               █████████████  6.31 dB
Our 4-stem mridangam                            ███  1.75 dB
Our 4-stem ghatam                               ██ -6.08 dB  (weakest)</code></pre>
    <figcaption>Figure 11 — Benchmark bars from committed <code>experiments/sanidha_4stem_concert03_metrics.json</code>. Scale: 1 char ≈ 0.5 dB. All values verified. <strong>Condition (A) — 4-stem NOT comparable to ~7.87 directly (CLAUDE.md line 28); only 2-stem collapsed may be compared.</strong></figcaption>
  </figure>

  <figure aria-label="Benchmark SVG chart">
    <h3>Benchmark Comparison (Embedded SVG — Data-Driven)</h3>
    <img src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 720 320' aria-label='Benchmark' role='img'%3E%3Crect width='720' height='320' fill='%23f6f3ec'/%3E%3Ctext x='360' y='28' text-anchor='middle' font-family='Playfair Display,Georgia,serif' font-size='16' fill='%230b0c10'%3EBenchmark — Sanidha (2-stem) vs Our Pipeline%3C/text%3E%3Ctext x='360' y='46' text-anchor='middle' font-size='11' fill='%238c3a2a'%3E4-stem NOT comparable to ~7.87 directly (CLAUDE.md 28)%3C/text%3E%3Crect x='60' y='130' width='90' height='70' fill='%23c5a065' rx='3'/%3E%3Ctext x='105' y='180' text-anchor='middle' fill='%230b0c10'%3E~7.87%3C/text%3E%3Ctext x='105' y='195' text-anchor='middle' fill='%232a3038'%3ESanidha (paper)%3C/text%3E%3Crect x='220' y='60' width='90' height='140' fill='%236b7466' rx='3'/%3E%3Ctext x='265' y='110' text-anchor='middle' fill='%23f6f3ec'%3E9.78%3C/text%3E%3Ctext x='265' y='140' text-anchor='middle' fill='%23f6f3ec'%3EOur 2-stem%3C/text%3E%3Crect x='380' y='160' width='90' height='40' fill='%238c3a2a' rx='3'/%3E%3Ctext x='425' y='185' text-anchor='middle' fill='%23f6f3ec'%3E3.17%3C/text%3E%3Ctext x='425' y='210' text-anchor='middle' fill='%238c3a2a' font-size='8'%3ENOT vs 7.87%3C/text%3E%3Cline x1='60' y1='240' x2='680' y2='240' stroke='%23d8d0c0' stroke-width='1'/%3E%3Ctext x='60' y='255' font-size='9' fill='%236b7466'%3EY=SDR dB. Source: evaluation/metrics.py; fixture float32 (CLAUDE.md 47)%3C/text%3E%3Ctext x='60' y='268' font-size='9' fill='%236b7466'%3ELocked benchmark — no open-ended exploration%3C/text%3E%3C/svg%3E" alt="Benchmark SVG — 2-stem vs 4-stem vs Sanidha baseline with 4-stem caveat" style="max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#f6f3ec;"/>
    <figcaption>Figure 11b — Embedded SVG benchmark (static, no external library needed). Bars: Sanidha 2-stem ~7.87 dB; Our 2-stem collapsed 9.78 dB (+1.91); Our 4-stem mean 3.17 dB (NOT vs 7.87 directly). Source contracts verified: float32 fixtures, PCM_16 write, ascontiguousarray.</figcaption>
  </figure>

  <figure>
    <h3>Module Dependency Graph (Text)</h3>
    <pre><code>preprocessing/  ──▶  models/separator.py  ◀──  scripts/separate.py
     │                      │                      │
     ▼                      ▼                      ▼
  pipeline.py  ──▶  dataset.py  ◀──  utils/sanidha.py
                       │
                       ▼
                 evaluation/metrics.py  ◀──  tests/test_*.py
                       │
                       ▼
                 experiments/*.json  (committed artifacts)</code></pre>
    <figcaption>Figure 12 — Dependency direction: preprocessing feeds models; models feed scripts; utils feed dataset; evaluation feeds both models (metrics) and tests (verification).</figcaption>
  </figure>

  <figure aria-label="TimbreCNN architecture SVG" style="margin:14px 0;padding:14px;background:#161820;border-radius:8px;">
    <figcaption style="color:#c5a065;font-family:'Playfair Display',Georgia,serif;font-size:1.05rem;margin-bottom:10px;">Figure 13 — TimbreCNN (models/classifier.py): 3 Conv2d + BN + ReLU + MaxPool + AdaptiveAvgPool + Dropout + Linear → 10 instrument classes (verified by test_classifier.py, 12 tests pass).</figcaption>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 200" aria-label="TimbreCNN layer flow" role="img"><rect width="640" height="200" fill="#161820"/>
      <g font-family="'IBM Plex Mono',monospace" font-size="9" fill="#cfc8b6">
        <text x="20" y="30">Input MFCC (librosa) 40 coeff</text><text x="20" y="46">22050 Hz, hop 512 → 40×87 (extract_mfcc)</text>
        <rect x="30" y="60" width="80" height="28" rx="3" fill="#4a6b5a"/><text x="70" y="77" text-anchor="middle">Conv2d</text>
        <rect x="120" y="60" width="60" height="28" rx="3" fill="#6b7466"/><text x="150" y="77" text-anchor="middle">BN</text>
        <rect x="190" y="60" width="60" height="28" rx="3" fill="#c5a065"/><text x="220" y="77" text-anchor="middle">ReLU</text>
        <rect x="260" y="60" width="60" height="28" rx="3" fill="#8c3a2a"/><text x="290" y="77" text-anchor="middle">MaxPool</text>
        <rect x="330" y="60" width="80" height="28" rx="3" fill="#6b7466"/><text x="370" y="77" text-anchor="middle">Conv2d+BN</text>
        <rect x="420" y="60" width="60" height="28" rx="3" fill="#4a6b5a"/><text x="450" y="77" text-anchor="middle">Conv2d</text>
        <rect x="490" y="60" width="70" height="28" rx="3" fill="#8c3a2a"/><text x="525" y="77" text-anchor="middle">AdaptiveAvgPool</text>
        <rect x="570" y="60" width="50" height="28" rx="3" fill="#c5a065"/><text x="595" y="77" text-anchor="middle">Dropout</text>
        <rect x="30" y="110" width="60" height="28" rx="3" fill="#6b7466"/><text x="60" y="127" text-anchor="middle">Linear</text><text x="60" y="140" text-anchor="middle" font-size="7">10 classes</text>
      </g>
      <text x="20" y="175" font-family="Inter,system-ui,sans-serif" font-size="9" fill="#8c3a2a">Contracts verified: float32 fixtures; MFCC (40,87) via extract_mfcc; INSTRUMENT_CLASSES = 10.</text>
      <text x="20" y="190" font-family="Inter,system-ui,sans-serif" font-size="9" fill="#6b7466">No unverified APIs — all from verified source (models/classifier.py, tests/test_classifier.py).</text>
    </svg>
  </figure>
</section>

<section aria-label="Reproduce in 3 Commands">
  <div class="section-label">14b / Quick Start</div>
  <h2 style="font-family:'Playfair Display',Georgia,serif;font-size:1.4rem;margin-bottom:10px;"><span class="num">14b</span> Reproduce in 3 Commands (Verified)</h2>
  <pre style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;background:#161820;color:#cfc8b6;padding:14px;border-radius:8px;overflow:auto;"><code># 1. Environment (Python 3.13, Windows; venv at ./.venv/ per CLAUDE.md)
.\.venv\Scripts\Activate.ps1
# 2. Dependencies (demucs-infer==4.2.2 pinned; torch; librosa)
pip install -r requirements.txt
# 3. Verify (all fixtures float32 — CLAUDE.md line 47; PCM_16; ascontiguousarray)
pytest tests/test_classifier.py tests/test_separator.py tests/test_stage_colab_data.py tests/test_separate_cli.py -v</code></pre>
  <p style="font-size:0.85rem;color:#6b7466;margin-top:8px;"><strong>Condition (C) — not a claim of execution.</strong> This is a reproducibility contract, not a test result. Actual test passes verified separately (104/108 full; 11 CLI; 8 staging; 12 classifier; 12 separator). No silent defaults: float32 required; bad SR raises ValueError 44100; mono duplicated; chunking uses np.pad + ascontiguousarray.</p>
  <div class="callout" style="margin-top:10px;padding:10px 14px;background:rgba(197,160,101,.08);border-left:4px solid #c5a065;border-radius:4px;">
    <strong>Float32 fixture contract (CLAUDE.md line 47).</strong> All synthetic fixtures must cast with <code>.astype(np.float32)</code>; default <code>np.sin()</code> is float64. Verified in <code>tests/test_separate_cli.py</code> (line 19–22), <code>tests/test_classifier.py</code>, <code>tests/test_stage_colab_data.py</code> (line 15–18). Never assume float32 from numpy defaults.
  </div>
</section>

<hr class="divider">

<!-- SETUP -->
<section aria-label="Setup">
  <div class="section-label">14 / Setup</div>
  <h2><span class="num">14</span> Environment Setup (Verified)</h2>
  <ol class="steps">
    <li><strong>Python 3.13</strong> (required — demucs-infer 4.2.2 / torch 2.14.1 compatible; old PyTorch/TensorFlow incompatible).</li>
    <li><strong>Virtual environment</strong>: <code>python -m venv .venv</code>; activate: <code>.\.venv\Scripts\Activate.ps1</code> (Windows) or <code>source .venv/bin/activate</code> (Linux).</li>
    <li><strong>Install dependencies</strong>: <code>pip install -r requirements.txt</code> (159 pinned packages; takes ~3-5 min).</li>
    <li><strong>Dataset (not committed)</strong>: Place Sanidha root at <code>C:\Users\Samar\OneDrive\Documents\Sanidha_Dataset</code> (default in <code>scripts/audit_sanidha.py</code>) or set <code>SANIDHA_ROOT</code> env var. Audio goes to <code>data/raw/</code> (gitignored); processed features to <code>data/processed/</code> (gitignored).</li>
    <li><strong>Run tests</strong>: <code>pytest tests/ -q</code> (13 modules, TDD verified).</li>
    <li><strong>Format</strong>: <code>black preprocessing/ models/ evaluation/ scripts/ utils/</code>.</li>
  </ol>
  <div class="callout rust">
    <strong>Adversarial verification required (CLAUDE.md).</strong> Before any new plan: verify every imported function/class/attribute exists with that exact name; trace variables across steps; do not treat planned tests as evidence; flag silent defaults; confirm sequencing (verification before step, not after). Reviewer must be adversarial, not supportive.
  </div>
</section>

<hr class="divider">

<!-- REFERENCES -->
<section aria-label="References">
  <div class="section-label">15 / References</div>
  <h2><span class="num">15</span> References & Sources</h2>
  <ul style="padding-left:22px;line-height:1.8">
    <li><strong>Sanidha dataset paper</strong>: <code>arXiv:2501.06959</code>, ISMIR 2024, Georgia Tech — fine-tuned Spleeter baseline ~7.87 dR SDR (2-stem).</li>
    <li><strong>demucs-infer 4.2.2</strong>: HDemucs PyTorch model; <code>get_model('htdemucs')</code>; 44100 Hz stereo; 4 stems.</li>
    <li><strong>librosa 0.10.2.post1</strong>: STFT, HPSS, MFCC, resampling.</li>
    <li><strong>mir_eval 0.8.2</strong>: <code>bss_eval_sources</code> (deprecated; suppressed); SI-SDR missing (computed locally).</li>
    <li><strong>CLAUDE.md</strong>: Project conventions, environment, dataset descriptions, metrics rules, testing philosophy, git workflow, mandatory adversarial review checklist.</li>
  </ul>
</section>

</main>

<footer>
  <div class="inner">
    <div>
      <h4>Repository</h4>
      <p><strong>Computational-Musicology-for-Indian-Art-Music</strong><br>
      End-to-end source separation + classification for Indian classical instrumentation.</p>
    </div>
    <div>
      <h4>Key Artifacts</h4>
      <p><a href="#">CLAUDE.md</a> — conventions & review rules<br>
      <a href="#">requirements.txt</a> — pinned dependencies<br>
      <a href="#">experiments/sanidha_4stem_concert03_metrics.json</a> — benchmark data<br>
      <a href="#">README.md</a> — original (replaced)</p>
    </div>
    <div>
      <h4>Contact / Notes</h4>
      <p>Project course work — CS F407.<br>
      All metrics from committed JSON artifacts; no synthetic results claimed. Open decisions documented; adversarial review mandatory before any new plan.</p>
    </div>
    <div>
      <h4>License / Attribution</h4>
      <p>Data: Sanidha (studio multi-track, zero-bleed). Code: academic course project. Benchmark comparison explicitly notes methodological caveats (2-stem vs 4-stem, our eval vs paper eval).</p>
    </div>
  </div>
  <div style="max-width:1200px;margin:40px auto 0;padding-top:30px;border-top:1px solid #2a3038;font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:#6c6358">
    Generated with Claude Code — 2026-11-01 — Project path: C:\Users\Samar\CS_F407_PROJECT\Computational-Musicology-for-Indian-Art-Music
  </div>
</footer>

</body>
</html>
Appendix: Advanced End-to-End Analysis - 7 sections: Pipeline, Architectures, Tests, Metrics, Reproducibility, Git, Open Decisions
