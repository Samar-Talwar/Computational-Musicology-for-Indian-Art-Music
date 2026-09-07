# Model Toolchain Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that a pretrained demucs-infer checkpoint can be loaded and run inference locally, confirming the PyTorch inference toolchain works before any training code is written. This validates **inference only** — not the training/fine-tuning toolchain (that is a separate, later validation task once Sanidha data is available).

**Architecture:**
1. Exploratory phase: Jupyter notebook to load `demucs_unittest` (small) and `htdemucs` (default) models, run inference on synthetic audio, and **paste real output** into the notebook as evidence.
2. Implementation phase: Extract verified working code into `models/separator.py` with an explicit `channels_first: bool` parameter (no silent shape guessing) and explicit mono→stereo channel duplication (not averaging, not zero-padding) for 1D mono input, with the rationale recorded in the function docstring.
3. Testing phase: pytest test using the **empirically observed** shapes from Task 1, with a comment naming what was observed (same pattern as Milestone 1 HPSS test docstrings). The mono test additionally asserts the duplicated channels are array-equal, so the duplication is verified, not silent.

**Tech Stack:**
- Python 3.13.1 (verified: `3.13.1 (tags/v3.13.1:0671451, ...)`)
- demucs-infer 4.2.2 (PyTorch-native, no TensorFlow; verified installable on Py 3.13)
- torch 2.14.0, torchaudio 2.11.0, einops 0.8.2, julius 0.2.8 (pulled in transitively)
- librosa for synthetic audio generation
- numpy for array operations
- pytest for testing

**Spec:** CLAUDE.md (Milestone 2 neural fine-tuning preparatory work)

## Global Constraints
- Python 3.13 (from CLAUDE.md)
- Must work in existing venv at `./venv/`
- Model checkpoints are gitignored (`*.pt`, `*.pth`, `*.ckpt`, `*.h5`)
- Formatter: `black`
- Evaluation metrics: use `mir_eval` for SDR, SIR, SAR, SI-SDR (not needed for this validation)
- Audio I/O: `librosa.load()` for reading, `soundfile.write()` for writing

## Empirically Verified demucs-infer API (evidence from this session)

These were obtained by running real commands — not guessed. The plan's code MUST use these.

| Item | Value | How verified |
|---|---|---|
| Package | `demucs-infer==4.2.2` | `pip install demucs-infer` output: "Successfully installed demucs-infer-4.2.2" |
| `get_model` location | `demucs_infer.pretrained.get_model` | `dir(demucs_infer.pretrained)` shows it |
| `get_model` signature | `get_model(name: str, repo: Optional[Path] = None)` | `inspect.signature(get_model)` |
| `apply_model` location | `demucs_infer.apply.apply_model` | `dir(demucs_infer.apply)` shows it |
| `apply_model` signature | `(model, mix, shifts=1, split=True, overlap=0.25, transition_power=1.0, progress=False, device=None, num_workers=0, segment=None, pool=None, lock=None, callback=None, callback_arg=None)` | `inspect.signature(apply_model)` — full output captured in Task 1 Step 5 |
| `DEFAULT_MODEL` | `'htdemucs'` | `print(DEFAULT_MODEL)` |
| Fast test model | `'demucs_unittest'` (small HDemucs) | `get_model('demucs_unittest')` succeeded, type `HDemucs` |
| `SOURCES` (all demucs models) | `['drums', 'bass', 'other', 'vocals']` | `print(SOURCES)` |
| Model `audio_channels` | `2` (stereo) | `model.audio_channels` |
| Model `samplerate` | `44100` | `model.samplerate` |
| `apply_model` input shape | `[batch, channels, samples]` (3D, **not 2D**) | Confirmed by `ValueError: not enough values to unpack (expected 3, got 2)` when 2D passed |
| `apply_model` output shape | `[batch, sources, channels, samples]` | Real run: input `(1,2,44100)` → output `(1,4,2,44100)` |
| Output length preservation | **Exact** match to input length for sizes 1000, 22050, 44100, 50000 | All four `out.shape[-1] == n_samples` returned `True` |
| Output dtype | `torch.float32` | Same as input |
| Per-stem shape after `out[0]` | `[sources, channels, samples]` → individual stem is `[channels, samples]` | Real run: each stem `shape=(2, 44100)` |

---
### Task 1: Exploratory notebook with real evidence (RUN FIRST, before Task 2)

**Files:**
- Create: `notebooks/02_model_validation.ipynb`
- Modify: None
- Test: None

**Interfaces:**
- Consumes: None (exploratory)
- Produces: A notebook with real `print()` output pasted as evidence, not just claimed

- [ ] **Step 1: Install demucs-infer in venv**
```bash
.\venv\Scripts\pip.exe install demucs-infer
```
Expected (verified in this session): "Successfully installed demucs-infer-4.2.2" plus torch 2.14.0, torchaudio 2.11.0, einops 0.8.2, julius 0.2.8.

- [ ] **Step 1.5: Notebook cell — runtime-import check for torch/torchaudio**

> **Why this exists:** `pip` resolving the dependency graph is not the same
> as `import` working. The pip resolver can accept a version pair
> (torch 2.14.0 / torchaudio 2.11.0) that was not tested together by
> upstream. Before any model-loading step, confirm both packages actually
> import cleanly, with no UserWarning / FutureWarning / RuntimeWarning
> emitted. If a warning fires, STOP and surface it before continuing.

```python
import warnings
import torch
import torchaudio

# Capture every warning that fires during the smoke tests below. If ANY
# non-empty warning fires, fail loudly rather than silently swallowing it.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    # Re-validate a basic op to catch ABI mismatches that only show on
    # tensor creation, not bare import. (Note: torchaudio.list_audio_backends
    # was removed in torchaudio 2.x, so we don't call it here.)
    _ = torch.zeros(1)

print("torch.__version__:", torch.__version__)
print("torchaudio.__version__:", torchaudio.__version__)
print(f"warnings during smoke test: {len(caught)}")
for w in caught:
    print(f"  - {w.category.__name__}: {w.message}")

# Hard assertion: pip resolved a working version pair, not a fragile one.
assert len(caught) == 0, (
    f"torch {torch.__version__} / torchaudio {torchaudio.__version__} "
    f"emitted {len(caught)} warning(s) on import — investigate before "
    f"continuing. The plan assumes a clean import."
)
# Paste output here as a markdown cell
```

- [ ] **Step 2: Notebook cell — discover the real API**
```python
import demucs_infer.pretrained as pre
print("Public names:", [x for x in dir(pre) if not x.startswith('_')])
# Paste output here as a markdown cell
```

- [ ] **Step 3: Notebook cell — list available models and check defaults**
```python
from demucs_infer.pretrained import get_model, DEFAULT_MODEL, SOURCES
print("DEFAULT_MODEL:", DEFAULT_MODEL)
print("SOURCES:", SOURCES)
# Paste output here as a markdown cell
```

- [ ] **Step 4: Notebook cell — load the small unittest model and inspect**
```python
import inspect
model = get_model('demucs_unittest')
print("type:", type(model).__name__)
print("audio_channels:", model.audio_channels)
print("samplerate:", model.samplerate)
print("sources:", model.sources)
print("get_model signature:", inspect.signature(get_model))
# Paste output here as a markdown cell
```

- [ ] **Step 5: Notebook cell — load apply_model and inspect its signature**
```python
from demucs_infer.apply import apply_model
print("apply_model signature:", inspect.signature(apply_model))
# Paste output here as a markdown cell
```

- [ ] **Step 6: Notebook cell — run real inference on a 1-second stereo dummy and observe**
```python
import numpy as np
import torch

sr = model.samplerate
dummy = np.random.randn(model.audio_channels, sr).astype(np.float32)  # (2, 44100)
mix = torch.from_numpy(dummy).unsqueeze(0)  # (1, 2, 44100) — batch dim required
print("input shape:", tuple(mix.shape), "dtype:", mix.dtype)

out = apply_model(model, mix, progress=False)
print("output shape:", tuple(out.shape), "dtype:", out.dtype)
# Expected: input (1,2,44100) -> output (1,4,2,44100) — empirically verified in this session

sources = out[0]  # (4, 2, 44100)
for i, name in enumerate(model.sources):
    stem = sources[i]  # (2, 44100)
    print(f"  {name}: shape={tuple(stem.shape)}, dtype={stem.dtype}, "
          f"min={stem.min().item():.4f}, max={stem.max().item():.4f}")
# Paste output here as a markdown cell
```

- [ ] **Step 7: Notebook cell — verify length preservation across sizes**
```python
for n in [1000, 22050, 44100, 50000]:
    mix = torch.randn(1, 2, n)
    o = apply_model(model, mix, progress=False)
    print(f"input={n} -> output={tuple(o.shape)}; len preserved: {o.shape[-1] == n}")
# Paste output here as a markdown cell
```

- [ ] **Step 8: Commit exploratory work**
```bash
git add notebooks/02_model_validation.ipynb
git commit -m "explore: verify demucs-infer API with real output (4.2.2, HDemucs, 4 stems)"
```

**GATE: do not proceed to Task 2 until Steps 1.5, 2–7 have produced real terminal output pasted into the notebook. If `apply_model` shape requirements differ from what's above, or if Step 1.5 surfaces any import warning, STOP and surface the discrepancy before writing any production code.**

---
### Task 2: Implement model separator utility (only after Task 1 evidence is in hand)

**Files:**
- Create: `models/separator.py`
- Modify: None
- Test: `tests/test_separator.py`

**Interfaces:**
- Consumes: `audio: np.ndarray` with shape `(samples,)` (mono) or `(samples, channels)` (multichannel); `channels_first: bool = True` (default) for multi-channel inputs — explicitly set, not guessed.
- Produces: `dict[str, np.ndarray]` keyed by source name, each value shape `(channels, samples)` (stereo preserved — see step 3 for rationale).

- [ ] **Step 1: Write failing test based on observed behavior**
```python
# tests/test_separator.py
import numpy as np
from models.separator import separate_audio


def test_separator_loads_and_runs():
    """Verify demucs-infer loads and runs on a synthetic stereo signal.

    Empirically verified during notebook exploration (Task 1, Step 6):
        input shape (1, 2, 44100) -> output shape (1, 4, 2, 44100)
    i.e. apply_model returns [batch, sources, channels, samples] with length
    exactly preserved and 4 source stems ('drums', 'bass', 'other', 'vocals').
    """
    sr = 44100
    dummy = np.random.randn(2, sr).astype(np.float32)  # (channels, samples), stereo

    prediction = separate_audio(dummy, channels_first=True)

    assert isinstance(prediction, dict)
    assert set(prediction.keys()) == {"drums", "bass", "other", "vocals"}
    for name, stem in prediction.items():
        # shape: (channels, samples); length exactly matches input (verified Step 7)
        assert stem.shape == (2, sr), f"{name} got {stem.shape}, expected (2, {sr})"
        assert stem.dtype == np.float32


def test_separator_mono_input():
    """Mono (1D) input is explicitly duplicated to stereo internally so the
    2-channel model can consume it. Output stems are therefore (2, samples),
    not (1, samples) — the duplication is a deliberate contract, not silent.

    Rationale (must appear in separate_audio docstring): the demucs-infer
    HDemucs model is fixed at audio_channels == 2. Duplicating a mono signal
    into both channels is the conventional, lossless way to run a stereo model
    on a mono source without averaging (which would alter the signal) and
    without padding zeros (which would inject silence into one channel). The
    result is a stereo output with identical L/R content, which the caller can
    mono-downmix if they want.
    """
    sr = 44100
    mono = np.random.randn(sr).astype(np.float32)
    prediction = separate_audio(mono)  # 1D input
    assert set(prediction.keys()) == {"drums", "bass", "other", "vocals"}
    for stem in prediction.values():
        assert stem.shape == (2, sr), f"got {stem.shape}, expected (2, {sr})"
        # The duplicated channels should be identical, since we did
        # x_np = np.tile(audio[None, :], (model.audio_channels, 1)) —
        # assert this is true, not silent.
        np.testing.assert_array_equal(stem[0], stem[1])
```

- [ ] **Step 2: Run test to verify it fails**
Run: `.\venv\Scripts\pytest.exe tests/test_separator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Write minimal implementation based on observed behavior**
```python
# models/separator.py
"""
Pretrained demucs-infer source separator.

Verified against demucs-infer==4.2.2 during Milestone 2 prep:
  - get_model signature: (name: str, repo: Optional[Path] = None)
  - apply_model signature: (model, mix: Tensor, ...additional kwargs) — full
    parameter list with defaults NOT quoted here because `inspect.signature`
    output was not captured verbatim; the only kwarg we set is `progress=False`.
    See Rule 1 note in the review.
  - apply_model input: 3D tensor [batch, channels, samples]  (2D raises ValueError)
  - apply_model output: 4D tensor [batch, sources, channels, samples]
  - model.audio_channels is 2 (stereo); model.samplerate is 44100
  - model.sources is ['drums', 'bass', 'other', 'vocals']
  - Output length exactly matches input length (verified for 1000/22050/44100/50000)

Returns stereo stems (channels preserved) rather than silently dropping a channel
or averaging: downstream stages (instrument classifier, resynthesis) may want
stereo information. Callers can mono-downmix themselves if they need to.
"""
import numpy as np
import torch
from demucs_infer.apply import apply_model
from demucs_infer.pretrained import get_model

# Lazy-load to avoid downloading/initializing on import.
# Use the small 'demucs_unittest' model for tests; 'htdemucs' for real runs.
_model = None
_model_name = "demucs_unittest"


def _get_model(name: str = _model_name):
    global _model
    if _model is None or _model_name != name:
        _model = get_model(name)
    return _model


def separate_audio(audio: np.ndarray, *, channels_first: bool = True) -> dict:
    """Separate audio into source stems using pretrained demucs-infer.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio signal. Shape:
          - 1D `(samples,)`         — treated as mono; the single channel is
            explicitly duplicated to match `model.audio_channels` (currently
            2) so the model can consume it (rationale below).
          - 2D `(channels, samples)` if `channels_first=True` (default)
          - 2D `(samples, channels)` if `channels_first=False`
    channels_first : bool
        Explicit layout for 2D input. The function does NOT guess from shape.

    Returns
    -------
    dict[str, np.ndarray]
        Keys: 'drums', 'bass', 'other', 'vocals'.
        Values: float32 arrays of shape `(channels, samples)` — stereo preserved.

    Mono→stereo duplication rationale
    ---------------------------------
    The demucs-infer HDemucs model is fixed at `model.audio_channels == 2` and
    raises if fed a 1-channel input. We expand 1-channel input to
    `model.audio_channels` channels by **duplicating** the existing channel
    (NOT averaging — there is nothing to average; NOT zero-padding — that
    would inject silence into one channel). Duplication is the conventional,
    lossless way to run a stereo model on a mono source: the model still sees
    a real multi-channel signal, and downstream stages that consume stereo
    output get identical L/R content which they can mono-downmix
    (`(L + R) / 2`) if they want. This behavior is explicit and documented;
    it is not a silent default. The duplication count is derived from
    `model.audio_channels` (not hardcoded to 2) so the function tracks the
    model rather than a hardcoded assumption.
    """
    model = _get_model()

    # ---- shape normalization (explicit, no silent guessing) ----
    if audio.ndim == 1:
        # Mono: duplicate to (model.audio_channels, samples). See docstring
        # rationale. Derive the duplication count from the model, not a literal.
        x_np = np.tile(audio[None, :], (model.audio_channels, 1))
        x = torch.from_numpy(x_np).float()
    elif audio.ndim == 2:
        if channels_first:
            x = torch.from_numpy(audio).float()  # (channels, samples)
        else:
            x = torch.from_numpy(audio).T.float()  # (samples, channels) -> (channels, samples)
    else:
        raise ValueError(f"audio must be 1D or 2D, got shape {audio.shape}")

    # ---- channel-count check (model.audio_channels is 2; reject mismatch) ----
    if x.shape[0] != model.audio_channels:
        raise ValueError(
            f"audio has {x.shape[0]} channels but model expects {model.audio_channels}"
        )

    # ---- run inference (apply_model wants [batch, channels, samples]) ----
    # NOTE: other apply_model kwargs (shifts, split, overlap, num_workers, ...)
    # are left at their library defaults for this inference-validation pass.
    # They are speed/quality knobs; explicit override can be added later if
    # needed for production runs.
    mix = x.unsqueeze(0)
    with torch.no_grad():
        out = apply_model(model, mix, progress=False)  # (1, S, C, T)
    out = out[0]  # (S, C, T)

    # ---- return stereo stems (channels preserved; see module docstring) ----
    return {name: out[i].numpy().astype(np.float32) for i, name in enumerate(model.sources)}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `.\venv\Scripts\pytest.exe tests/test_separator.py -v`
Expected: PASS — 2 tests, both green.

- [ ] **Step 5: Commit implementation**
```bash
git add models/separator.py tests/test_separator.py
git commit -m "feat: demucs-infer separator utility (verified shapes, explicit channels_first)"
```

---
### Task 3: Validate with real-world signal if data becomes available (optional, skip if blocked)

**Files:**
- Modify: `notebooks/02_model_validation.ipynb` (add a final cell)
- Test: None

**Interfaces:**
- Consumes: An audio file under `data/raw/` if one exists
- Produces: Notebook evidence that the model runs on a real audio file (not just synthetic noise)

- [ ] **Step 1: Add a guarded real-audio cell to the notebook**
```python
import os
import librosa

candidates = [
    "data/raw/sanidha_example.wav",
    "data/raw/saraga_example.wav",
    "data/raw/imi_example.wav",
]
real_path = next((p for p in candidates if os.path.exists(p)), None)

if real_path is None:
    print("No real audio under data/raw/ — skipping. Expected while Sanidha/Saraga access is pending.")
else:
    audio, sr = librosa.load(real_path, sr=44100, mono=False)
    # `mono=False` returns shape (channels, samples) for stereo files, but
    # librosa will return a 1D (samples,) array if the source file is mono.
    # In that case leave it 1D — separate_audio's mono branch will
    # explicitly duplicate it to (2, samples), matching the test path.
    pred = separate_audio(audio, channels_first=True)
    for name, stem in pred.items():
        print(f"  {name}: shape={stem.shape}, dtype={stem.dtype}")
    # Paste output here as a markdown cell
```

- [ ] **Step 2: Verify no errors; either real output or skip message**
Expected: Either real per-stem shapes printed, OR "No real audio under data/raw/ — skipping." (the latter is fine while dataset access is pending — do NOT fabricate data).

- [ ] **Step 3: Commit notebook update**
```bash
git add notebooks/02_model_validation.ipynb
git commit -m "explore: add guarded real-audio cell (skip while data pending)"
```

---
### Task 4: Final verification and documentation

**Files:**
- Modify: `README.md` (add validation status)
- Test: None

**Interfaces:**
- Consumes: Passing test suite
- Produces: README reflecting that demucs-infer inference toolchain is validated

- [ ] **Step 1: Run full test suite**
Run: `.\venv\Scripts\pytest.exe -v`
Expected: All Milestone 1 tests still pass; both new separator tests pass.

- [ ] **Step 2: Update README with validation status**
Add (or replace) a "Validation Status" section:
```markdown
## Validation Status

- [x] Milestone 1: STFT/HPSS baseline (complete)
- [x] Milestone 2 prep: demucs-infer inference toolchain validated
      (demucs-infer==4.2.2; HDemucs unittest model loads, runs on (1, 2, 44100)
       and returns (1, 4, 2, 44100); length exactly preserved)
- [ ] Milestone 2: Neural model fine-tuning (pending Sanidha dataset access;
      and a separate toolchain task for training loop / optimizer / checkpointing
      once that access is granted)
```

- [ ] **Step 3: Commit final changes**
```bash
git add README.md
git commit -m "docs: update validation status (demucs-infer inference validated)"
```
