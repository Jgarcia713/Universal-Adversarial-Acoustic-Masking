# Acoustic Keystroke Masking — Project Overview and Technical Handoff

**Status:** working digital universal-perturbation proof of concept (17 October 2026)  
**Audience:** project teammates and AI coding assistants. **Read this file before changing data loading, preprocessing, models, or evaluation.**

## 1. Project goal and threat model

Acoustic side-channel classifiers can infer which keyboard key was pressed from short microphone recordings. We are exploring a **defense**: play a reusable masking sound so that keystrokes become harder to distinguish. We first test digitally mixed waveforms, then want to investigate multiple masking patterns, adaptive attackers who learn the patterns, environmental acceptability, and eventually playback/recording effects.

- **Defender's control:** a masking waveform `delta`, added to or played alongside keystrokes. The current prototype uses **digital addition** of a single shared 0.25-second waveform, aligned to the beginning of each already-cropped keystroke. This assumes timing information that would not automatically be available in a real deployment.
- **Attacker's goal:** infer the **original pressed key**, not identify which masking pattern is present. A stronger attacker may retrain on perturbed sounds.
- **What has NOT been demonstrated:** speaker-to-microphone transfer, unknown keypress alignment, generalization to new keyboards/rooms, naturalness, resilience against an adaptively trained classifier, or privacy protection in real-world conditions.

## 2. How we selected the model

1. We investigated *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards* (often called DeepKeyAttack). Its CoAtNet-style classifier and mel-spectrogram input are relevant, but reproducing that relatively heavyweight training procedure would slow work on our actual contribution: audio masking.
2. We then studied the 2026 paper *Automated Acoustic Side-Channel Attack on Keyboard Inputs via Combined Video–Audio Analysis* and its [authors' implementation](https://github.com/k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack). Its [dataset repository](https://github.com/k3rnel-pan1c-ksd/input-sounds) supplies pre-cropped keystroke WAV files. Despite the title, **video/OCR helps label recordings; the classifier predicts from audio features**.
3. A saved `weights.best.basic_cnn.keras` model was loadable with legacy Keras and had input `(64,25,1)`, 47 output units, and ~198,575 parameters. But both inspected WAV datasets use 0.25-second clips and the authors' current feature settings yield `(64,50)`; the base dataset contains **67** named key labels, the transfer dataset **68** (`rcontrol` is the extra). We could not establish the old checkpoint's exact preprocessing or 47-class label order. Forcing the input to 25 frames gave predictions but did **not** validate correctness. **Do not use this undocumented checkpoint as a trusted baseline.**
4. We trained our own small modern TensorFlow/Keras CNN on **26 letters (`a`–`z`)** from the base dataset. This unlocked the UAP code and created a known class mapping and reproducible pipeline.
5. **Next:** a teammate should train the stronger **factored CNN** in `ZnanstveniRad_final.ipynb` and integrate it with our shared 26-letter interface (see `MODEL_TRAINING_GUIDE.md`). CoAtNet is optional, not a prerequisite.

## 3. Audio basics: from a `.wav` to a CNN input

### 3.1 Raw audio and sampling

A WAV file contains a sequence of amplitude measurements. At **44,100 samples/second**, a **0.25-second** keystroke clip contains **11,025 samples**. Once decoded, our waveform array has shape `[11025]` and typically uses floating-point values around `[-1,1]`. Every file is read as mono and is padded/truncated to 11,025 samples; files with different sampling rates are resampled to 44,100 Hz.

Filename examples: `a_1.wav`, `a_125.wav`, `z_42.wav`; the filename prefix is the **true key label**. The numeric suffix is a recording identifier, **not a class**.

### 3.2 What is a spectrogram?

A waveform shows **how loud the air-pressure signal is over time**, but many keys differ in frequency content and how their clicks decay. A spectrogram answers: **which frequencies have energy at each moment?**

1. **STFT (short-time Fourier transform):** slide a 1,024-sample window across the 11,025-sample waveform, moving **225 samples** per frame. Fourier analysis estimates the strength of different frequencies inside each window. Using centered frames, the result has **50 time frames** and **513 frequency bins** (positive-frequency bins for a real-valued 1,024-point FFT).
2. **Power:** take `abs(STFT) ** 2`. A power spectrogram represents energy per time/frequency bin.
3. **Mel filter bank:** combine 513 linear-frequency bins into **64 overlapping mel-frequency bands**. Mel spacing roughly reflects human frequency perception and reduces dimensionality. These filters are weighted averages, **not a second Fourier transform**.
4. **Decibels:** `librosa.power_to_db(mel, ref=np.max)` converts power to logarithmic dB relative to the **maximum mel power in that clip**. Thus each clip's loudest mel bin becomes about 0 dB; low-energy values are negative, with librosa's default floor at approximately -80 dB relative to the maximum. The relative normalization makes the representation less sensitive to overall volume, but does not remove every amplitude effect.
5. **CNN input:** the resulting matrix has shape `[64 mel bands, 50 time frames]`; add one channel dimension → `[64,50,1]`; stack samples → `[batch,64,50,1]`.

Think of the log-mel spectrogram as a small grayscale image with **frequency vertically** and **time horizontally**. A CNN can detect repeatable acoustic patterns in that image. It does not interpret the pixels as written characters; it learns correlations between sound patterns and the key labels.

### 3.3 Exact preprocessing contract — DO NOT silently change

```python
SAMPLE_RATE = 44100
N_SAMPLES = 11025
N_FFT = 1024
HOP_LENGTH = 225
N_MELS = 64
CLASSES = list('abcdefghijklmnopqrstuvwxyz')

import librosa
import numpy as np

def extract_features(waveform):
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        center=True, pad_mode='constant'
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
```

**Crucial debugging discovery:** We initially used `tf.pad(..., mode='REFLECT')` for differentiable preprocessing, but `librosa.feature.melspectrogram` in the environment used **constant (zero) padding**. Spectrogram error was mean **0.3355 dB**, maximum **33.6054 dB**; with `tf.pad(..., mode='CONSTANT')`, error dropped to mean **5.28e-6 dB**, maximum **1.60e-4 dB** on eight test clips. Therefore, both offline and differentiable preprocessing must explicitly use **zero padding**. There is **no reason to retrain** the existing small model to make this default explicit.

For differentiable attacks, implement centered `tf.signal.stft` after padding both sides by 512 zeros, periodic Hann window, squared magnitude, and a fixed filter bank made with `librosa.filters.mel(sr=44100,n_fft=1024,n_mels=64)`. Apply the per-example log-power normalization and default 80 dB floor, then transpose `[batch,frames,mels]` to `[batch,mels,frames,1]`. A gradient check produced a nonzero finite waveform gradient (norm ~1.13 for one sample). Verify numerical agreement and prediction agreement again if libraries or implementations change.

**Never preprocess perturbed audio with librosa inside `GradientTape`:** librosa/NumPy conversion interrupts TensorFlow's gradient path. Use the verified `tf_log_mel()` implementation during perturbation training; offline librosa processing remains a useful independent evaluation reference.

## 4. Current implementation

### Data
- Source folder: `input-sounds/cropped_base_pretraining`, using lowercase `a`–`z` only, up to 125 WAVs per key.
- Each waveform is float32 `[11025]`; labels are integer indices `a=0` through `z=25`.
- A stratified **70/15/15** train/validation/test split with seed **42** is saved as explicit indices. The final test set used in the reported experiment held **488 recordings**.
- Test clips are held out from **the small model's training** and **UAP optimization**, but come from similar/same-session recording conditions. This is a narrow evaluation.

### Baseline classifier
A small 3-convolution CNN was built in current TensorFlow/Keras: `Conv2D(32) → BN → pool → Conv2D(64) → BN → pool → dropout → Conv2D(128) → BN → pool → dropout → flatten → Dense(128) → BN → dropout → Dense(26, softmax)`. Input `[64,50,1]`. Trained with Adam (`1e-3`), sparse categorical cross-entropy, batch size 32, early stopping, up to 30 epochs. Validation and test accuracy were approximately **100% within this dataset**, not evidence of cross-session robustness.

### Universal waveform perturbation
We freeze the CNN and optimize a **single shared** waveform `delta` with shape `[11025]`. For each training batch:

```text
keystroke waveform x + same delta
    → clip result to [-1,1]
    → differentiable TensorFlow log-mel
    → frozen CNN softmax
    → sparse cross-entropy on ORIGINAL key label y
    → gradient ascent on delta
    → clamp delta samples to [-epsilon,+epsilon]
```

Initial run: **untargeted** projected sign-gradient ascent, `epsilon=0.01`, step size `0.001`, 30 epochs, batch size 32. The loss intentionally rises so true-key classification worsens. Current experiment aligned `delta[0]` to the start of every cropped audio clip; real-time playback alignment is unsolved.

### Observed results (initial digital test)

| Condition | Top-1 accuracy on 488 held-out clips |
|---|---:|
| Clean | 100.00% |
| One uniform random waveform, `[-0.01,+0.01]` | 67.21% |
| Learned UAP | **3.89%** |

- Clean cross-entropy: **0.00388**; UAP cross-entropy: **263.56** (verify any future loss implementation before comparing magnitudes).
- Learned UAP peak amplitude: **0.0100**; RMS: **0.0079889**.
- **All 488 perturbed clips were classified as the letter `b`**. The 3.89% accuracy reflects the percentage of actual `b` examples, **not random/uniform predictions**. This is *untargeted training resulting in prediction collapse*; we did not set a `b` target.
- Uniform `[-epsilon,epsilon]` noise has a smaller expected RMS (`epsilon/sqrt(3)` ≈ 0.00577). We have **not yet reported a properly controlled RMS-matched and peak-constrained multi-seed baseline**. A scaled Gaussian control produced peak ~0.03138, violating `epsilon=0.01`; do not report it as an equal-constraint comparison.
- This is evidence of vulnerability for this **within-session, digitally aligned, surrogate CNN** only. It does not establish a deployable masking defense.

## 5. Shared files and interfaces

The current Python notebook saves these artifacts (confirm which files have actually been committed or uploaded before using):

| Artifact | Contents / contract |
|---|---|
| `keystroke_cnn.keras` | Saved small classifier, `(None,64,50,1)` → `(None,26)` softmax |
| `keystroke_dataset.npz` | `waveforms`, `labels`, `filenames`, `train_idx`, `val_idx`, `test_idx` |
| `preprocessing_config.json` | `classes`, `sample_rate`, `n_samples`, `n_fft`, `hop_length`, `n_mels`, `input_shape`, etc. |
| `uap_epsilon_001.npy` | Learned shared float32 perturbation `[11025]` |
| `uap_epsilon_001.wav` | Optional 44.1 kHz float WAV export of the same perturbation |

The `.npz` is a compressed set of arrays; keeping split indices prevents accidental changes in train/test membership. The `.json` records the **meaning** of the arrays and model outputs. The WAV/NPY perturbation contains **no key label**: it is universal. A key label belongs to the underlying keystroke recording.

Expected loading:

```python
import json, numpy as np
from tensorflow import keras
model = keras.models.load_model('keystroke_cnn.keras', compile=False)
with np.load('keystroke_dataset.npz', allow_pickle=False) as z:
    waveforms, labels = z['waveforms'], z['labels']
    train_idx, val_idx, test_idx = z['train_idx'], z['val_idx'], z['test_idx']
with open('preprocessing_config.json') as f:
    cfg = json.load(f)
delta = np.load('uap_epsilon_001.npy').astype(np.float32)
assert model.output_shape[-1] == len(cfg['classes']) == 26
assert len(delta) == cfg['n_samples'] == waveforms.shape[1]
```

For future models, use this interface unless a deliberate, documented migration is agreed upon. Most importantly, keep **class order identical**.

## 6. Work division and near-term priorities

- **Waveform/UAP owner:** characterize the current collapse, evaluate better noise baselines, sweep perturbation strength, build multiple masking patterns and environmental loss, measure robustness to misalignment and physical playback.
- **Stronger-model owner:** train and evaluate the authors' factored CNN with the **same 26 classes and preprocessing**, export compatible artifacts. See `MODEL_TRAINING_GUIDE.md`.
- **Adaptive-attacker owner:** retrain a classifier on **original labels with clean and masked audio**, ensuring no test leakage, then test held-out perturbations and generalization. See `ADAPTIVE_ATTACKER_GUIDE.md`.

Open questions: cross-session test accuracy, strong-model clean baseline, mask amplitude/loudness tradeoff, effectiveness after adaptive training, mask pattern diversity, environmental plausibility, and physical speaker/mic robustness.

## 7. Sources

- [Authors' model notebook repository](https://github.com/k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack) — `ZnanstveniRad_final.ipynb`, plus comparison/other notebooks.
- [Authors' labeled recordings repository](https://github.com/k3rnel-pan1c-ksd/input-sounds) — `cropped_base_pretraining`, `cropped_transfer_learning`, saved older checkpoint.
- Authors' paper: *Automated Acoustic Side-Channel Attack on Keyboard Inputs via Combined Video–Audio Analysis* (2026).
- Earlier motivating work: *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards* (DeepKeyAttack, 2023).

**For AI assistants:** treat current metrics as reported experiment outputs, not guarantees; distinguish implemented code from recommended future experiments; never infer key class order from a model's output size; never change the label mapping/padding silently; do not leak test examples into training, UAP optimization, hyperparameter selection, or adaptive training.
