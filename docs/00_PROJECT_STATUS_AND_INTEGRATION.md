# Acoustic Keystroke Masking — Project Status and Integration Contract

**Audience:** all three teammates, reviewers, and coding agents.  **Time remaining:** approximately 3–4 weeks at handoff.  **Read this file before changing implementation.**

> This guide reflects the repository README supplied on November 20, 2026 and the team's reported experiments. It does **not** assert that the latest mask-bank implementation, directory names, or branch structure were independently inspected. **Inspect `README.md`, `pyproject.toml`, `configs/`, `src/acoustic_masking/`, `tests/`, and the actual bank manifest in the checked-out main branch before coding.** Where this guide uses a proposed schema or API, it is explicitly labelled.

## 1. The research question and deliverable

**Goal:** measure whether adversarial audio residuals mixed with plausible environmental audio can reduce a classifier's ability to infer which letter was typed, and how much recovery is possible when an attacker learns from one or many masks.

Our *current threat model* is **digital waveform mixing**: a keystroke recording + a sampled rain/fan/wind carrier window + a short, phase-shifted residual. This is **not** yet a demonstrated physical speaker-to-microphone defense. Lower key-classification accuracy means a stronger defense *for a fixed evaluation condition*; higher accuracy means the attacker is recovering.

The final report should separate three questions:

1. **Effectiveness:** how much better are optimized masks than carrier-only/random-noise controls?
2. **Adaptive robustness:** what happens when an attacker trains with K different masks? Does recovery on known vs unseen masks differ?
3. **Practical trade-off:** how does reducing residual volume affect key-classification accuracy and perceived acceptability?

Finish controlled, reportable experiments. Avoid new architectures, psychoacoustic objectives, real-time deployment, and extensive tuning unless the essential tasks are complete.

## 2. What is already implemented and observed

### Existing repository capabilities (per supplied README)

- Package `src/acoustic_masking/` with CLI, classifier, data loading, `librosa`/TensorFlow log-mel preprocessing, audio operations, residual training, paired evaluation, artifact recording and plots.
- Historical **26-class CNN** and saved dataset in `artifacts/models/`.
- Carriers `rain.wav`, `fan.wav`, `wind.wav` in `artifacts/carriers/`.
- Saved original UAP and historical residual bank in `artifacts/masks/`; run-specific artifacts in `artifacts/runs/<run_name>/`.
- CLI commands: `python -m acoustic_masking --help`, `evaluate`, `plot-results`, `train-residual`, `prepare-dataset`, `train-classifier`, and `extract-saved-residual`. **Inspect current `--help`; commands may have changed since the README.**
- Regression tests: run `python -m pytest -q`; additionally `python -m pytest -q tests/test_preprocessing.py` for feature equivalence/gradients.

### Reported experimental findings — not all are final independent-test results

- Re-training a scratch multi-carrier residual yielded fan **44.47%** vs saved **44.16%**, rain **49.67%** vs **49.58%**, wind **53.92%** vs **54.02%**. These are close development-evaluation results.
- Eight-mask pilot: only **3/8** 180-epoch masks met a strict rule of within **2 percentage points** of the scratch reference on **each** carrier. All still helped compared with carrier-only conditions. Wind was limiting.
- Controlled replay of the same eight trajectories reproduced epoch-180 waveforms/histories exactly; checkpoints at **240** and **320** epochs admitted **8/8**. Average wind accuracy: at 180 epochs zero/random **57.75% / 56.68%**; at 240 **55.85% / 55.26%**; at 320 **54.39% / 54.28%**. Training at 320 epochs was selected for the main bank because the worst 240-vs-320 wind penalty was 1.56 pp.
- Four zero-start vs four random-start pilot mask pairs: *different seeds* generally had circular-aligned waveform correlation ~0.10 and prediction disagreement ~6%; same-seed/different-initialization pairs were more correlated (~0.37–0.45) and disagreed ~4.5–5%. **This particular table contains 28 comparisons among eight masks, not all 496 pairs of a 32-mask bank.**
- **32 candidates reportedly trained** at the chosen settings. Exact inventory, masks passing qualification, manifest paths, 24/8 split, and listening-test data must be verified from the actual merged repository. Do not claim the 32 are all qualified or frozen without checking.
- Preliminary listening: masks are usable research candidates, but fan/wind may require lower residual gain; no formal audibility/inaudibility result established.

**Major evidence limitation:** The nominal keystroke `test_idx` and carrier validation recordings were repeatedly inspected during defense development. For the paper, label their results **development evaluation**, even when the config says `test`. Thirty repeats change carrier windows and phases, **not independently recorded keystrokes**. A credible independent test set is highly desirable and should be identified by the defense owner; teammates should not claim generalization from old random-file splits alone.

## 3. Non-negotiable compatibility contract

| Item | Requirement |
| --- | --- |
| Task | 26-way **letter classification**, labels `0=a, 1=b, ..., 25=z`; no extra `masked` class |
| Keystroke audio | mono, `float32`, **44,100 Hz**, **11,025 samples** (0.25 s) |
| Model feature shape | `[batch, 64, 50, 1]` log-mel dB; confirm in test rather than assuming |
| FFT / hop / mel | `n_fft=1024`, `hop_length=225`, `n_mels=64` |
| STFT | `center=True`, **constant/zero padding**, Hann window as implemented by repo |
| Mel / dB | same repo preprocessing, `power_to_db(..., ref=np.max, top_db=80)` equivalence; do not replace with MFCC, global scaling, z-score, differing mel filter conventions |
| Dataset | load **prepared `.npz` via `src/acoustic_masking/data.py`**; preserve its exact split indices |
| Mask | **11,025 float32** samples; waveform is a perturbation, **not** a keystroke or labelled example |
| Ground truth | always the original keystroke `labels` (do not use the class the masked model mistakenly predicts) |
| Carriers | existing loader produces mono 44.1kHz, 100 Hz high-pass and chronological splits; preserve source split identity |
| Mixture | reuse repo implementation of carrier normalization, per-carrier target RMS, `residual_gain`, random carrier windows, circular phase shifts, playback scale and clipping policy |
| Baselines | keep original frozen CNN and published/historical residuals unchanged; save any new model/output under a **new** path |

### Prepared NPZ — inspect before consuming

As documented by the checked repository's original loader, `keystroke_dataset.npz` contains at least `waveforms`, `labels`, `train_idx`, `val_idx`, `test_idx`; it may additionally contain filenames and metadata. It does **not** contain spectrograms by default, nor does it encode masking labels.

```python
import numpy as np
from acoustic_masking.data import load_dataset

d = load_dataset('artifacts/models/keystroke_dataset.npz')
x = d['waveforms']        # expected [N, 11025], float32
labels = d['labels']      # expected [N], integer 0..25
train_idx, val_idx, test_idx = (d[k] for k in ('train_idx','val_idx','test_idx'))
assert x.shape[1] == 11025 and len(x) == len(labels)
assert set(train_idx).isdisjoint(val_idx) and set(train_idx).isdisjoint(test_idx)
assert set(val_idx).isdisjoint(test_idx)
```

If using raw `.npz` instead, always use `np.load(path, allow_pickle=False)` and inspect `npz.files` and shapes. `residual_bank.npz` is **a different format**: its keys are names of residual waveforms, **not** `waveforms`/`labels`/split indices. New bank manifests may reference per-run `.npy` files instead of putting all masks into one NPZ. Determine the format from the actual artifact and documented loader; **never assume an NPZ key.**

### Correct audio composition

Conceptually:

```python
# PSEUDOCODE; replace every operation with the repository's matching audio helpers.
# Select keystroke x and true label y from one permitted split.
carrier_window = sample_from_configured_carrier_split(...)
mask = sample_mask_from_selected_K(...)
shifted_mask = circular_shift(mask, phase_samples)
mixture = x + scaled_carrier_window + playback_scale * carrier_residual_gain * shifted_mask
# Match clip/no-clip convention EXACTLY for the comparison being run.
# Then call the canonical log-mel transform and predict original label y.
```

The README says **training clips mixtures**, whereas **historical evaluation uses `clip_mixtures: false`** to reproduce notebook results. This discrepancy must be stated and controlled. Use one documented policy for all *new* compared conditions; never mix different clipping policies within one comparison. Also carry across source splits and `carrier_rms` configuration. Do not silently normalize each 0.25-second carrier window independently if repository training normalizes the **whole carrier split** first.

### Preprocessing equivalence: tests to run before model work

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m pytest -q tests/test_preprocessing.py
```

Then confirm on **the same keystroke** that `librosa_log_mel` and the repo's TensorFlow `LogMel` agree numerically, including orientation `[64,50]`. Confirm waveform gradients are finite if differentiating. **Import the actual functions from `preprocessing.py`; do not copy an approximate function into a new file.**

## 4. Evaluation/data leakage rules

- Train classifier weights on `train_idx`. Tune architecture/hyperparameters on validation data. Do not select models based on the nominal test set.
- Mask optimization uses the keystroke training split; adaptive attacker training may use those masked training examples with **original labels**. Do not feed evaluation/test recordings into either optimizer.
- Reserve **unseen mask IDs** entirely out of adaptive attacker fitting, early stopping, and selection. Distinguish *unseen mask* from *unseen keystroke* and *unseen carrier recording*; these are three independent dimensions.
- Freeze mask qualification, subset assignments and comparison criteria **before** reviewing adaptive results. Selection of high-performing masks on development data does not make the selected bank an independent evaluation dataset.
- Paired comparisons must use identical keystrokes, carrier starts, residual phase draws, gain, preprocessing and conditions; changing only the intended variable.
- Record `model_id`, `model_hash`, dataset/split identity, mask IDs, bank manifest hash, configuration, RNG seeds, software/hardware, input policy, epoch/update budget, and exact number of evaluation examples. If a field cannot be recorded, document the gap.
- Avoid overclaiming statistical independence: repeated phases of the same clip are repeated measures, not new independent keystroke recordings. Prefer per-original-clip paired analysis and/or group-aware uncertainty intervals when possible.

## 5. Roles, priorities and expected handoffs

| Owner | Must deliver | Helpful if time | Defer |
| --- | --- | --- | --- |
| **Defense/masking owner** | Merge tested bank and manifests to main; small **fixed-gain sweep** (e.g. 0, −1, −2, −3 dB) with paired results for rain/fan/wind; consolidate baseline figures and find an independent keystroke test if feasible | concise structured listening observations, repeated baseline consistency checks | perceptual-loss optimization, dynamic volume, over-the-air testing |
| **Stronger-model teammate** | Compatible stronger 26-letter classifier; reproduce feature contract, clean baseline, report training/validation methodology and test results; integrate into shared evaluation | compare mask transfer and model differences; one independent-session dataset | broad architecture search, full legacy OCR pipeline |
| **Adaptive-attacker teammate** | K-versus-accuracy experiment using frozen bank, **seen/unseen** mask evaluation and fixed-budget comparison; export results and figure-ready CSV | more seeds, recovery-cost curves, test stronger model once available | complex adversarial co-training / dynamic mask synthesis |

### Suggested 3–4 week schedule

- **Week 1:** merge/handoff contracts; independent model and adaptive code baselines; test preprocessing and mask loader; defense owner starts report and gain sweep.
- **Week 2:** initial adaptive K results and stronger-model reproducibility; catch leakage/format issues; run one integration check together.
- **Week 3:** final fair comparisons, repeat key seeds, confidence/uncertainty analysis, tables and figures; write methods/results.
- **Week 4 (if available):** integration, report editing, limitations, presentation, final reproducibility pass. Freeze new scope early enough to finish writing.

## 6. How agents should work in this repo

1. Read `README.md`, this file, your role-specific guide, `configs/`, and **actual** implementation in `src/acoustic_masking/` before proposing an edit.
2. Inspect `python -m acoustic_masking --help`, data `.npz.files`, `.npy` shapes, bank manifest JSON, and existing tests. **A documentation example is not proof an API exists.**
3. State a short plan and list affected files. Make the smallest possible diff in a feature branch. Never alter historic experiment outputs in place.
4. Add or run tests for shapes, label mapping, split separation, mixing parity, deterministic mask sampling, and model compatibility.
5. Run a **smoke test on a tiny subset**, then a baseline comparison, before launching full training.
6. Save metrics, exact CLI command/config, artifacts and limitations. Distinguish observed values from expectations.
7. Open a PR documenting changed files, proof of interface compatibility, and **one command a teammate can run to reproduce a key result**. No merge of untested experiments.

## 7. Report vocabulary and caveats

- Say **"digital mixing, recorded carriers, fixed 26-letter dataset"** rather than real-world acoustic protection.
- Say **"development/nominal test split"** for historical paired results, not independent held-out evaluation.
- Treat near-perfect clean accuracy on a single recording corpus as a sign the dataset may be easy or session-specific, not proof of real-world generalization.
- Explain **known-mask vs unseen-mask** and **known-carrier recording vs independent-carrier recording** separately.
- Listening observations are preliminary; do not claim inaudibility or imperceptibility without controlled evidence.
- A 32-candidate bank is not automatically a 32-mask qualified bank. Report actual inventory and policy.
