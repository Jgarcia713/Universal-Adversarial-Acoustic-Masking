# Implementation Changelog — Acoustic Keystroke Masking

This file records **work actually completed and measured**, followed by a separate list of planned changes. Entries describe the project development sequence; exact commit dates were not recorded for all notebook experiments. Last documentation update: **2026-10-17**.

## Current state — 2026-10-17

**Working prototype:** 26-letter baseline CNN, reproducible saved dataset/configuration, validated differentiable log-mel preprocessing, and one digitally applied universal adversarial waveform (`epsilon=0.01`). Within-session 488-sample test: 100% clean accuracy, 67.21% with an initial uniform-noise control, **3.89% under learned UAP**. All 488 UAP predictions collapsed to `b`. Stronger CNN, adaptive training, multiple masks, naturalness constraints, and physical tests are **planned, not implemented**.

## History (development order)

### 1. Project scope and literature survey

- Defined goal: speaker-based masking of acoustic keystrokes to reduce information recoverable by microphone-based classifiers.
- Investigated DeepKeyAttack/CoAtNet-style acoustic classification. Decided a heavyweight reimplementation would delay waveform attack/defense development.
- Identified *Automated Acoustic Side-Channel Attack on Keyboard Inputs via Combined Video–Audio Analysis* and its `ZnanstveniRad_final.ipynb` implementation and `input-sounds` WAV repository.
- Distinguished OCR-assisted **label creation** from audio-only keystroke **inference**.
- Allocated future stronger-model work to teammate; retained lightweight classifier as development baseline.

### 2. Attempted to reuse upstream checkpoint

- Loaded `saved_models/model_1/weights.best.basic_cnn.keras` with legacy Keras.
- Found input `(64,25,1)`, 47 output classes, ~198,575 parameters; initial paper feature extraction from 0.25s/44.1 kHz audio with `n_fft=1024`, `hop_length=225`, `n_mels=64` yields `(64,50)` instead.
- Tested `hop_length=450` and resized spectrograms. Both yielded output index 36 for an example, with differing confidence, but original label mapping remained unknown.
- Counted dataset label names: **67** in `cropped_base_pretraining`; **68** in `cropped_transfer_learning` (additional `rcontrol`). Neither directly matches 47-output saved model.
- **Decision:** do not trust the old checkpoint; train a new, documented CNN.

### 3. Trained baseline 26-letter CNN

- Selected lowercase `a`–`z`; loaded `key_1.wav` through `key_125.wav` where available, mono, 44.1 kHz, 0.25s / 11,025 samples.
- Implemented librosa 64-band log-mel preprocessing (`n_fft=1024`, `hop_length=225`, per-clip `power_to_db(ref=np.max)`) yielding `(64,50,1)`.
- Generated stratified 70% train / 15% validation / 15% test indices (seed 42).
- Implemented a 3-convolution + 128-dense-layer Keras classifier with 26-way softmax. Trained with Adam and early stopping; observed nearly 100% validation and 100% test accuracy on same-session-style split. **No external generalization claim.**
- Saved `keystroke_cnn.keras`, `keystroke_dataset.npz`, and `preprocessing_config.json`; implemented reloading and inference consistency checks.

### 4. Built and validated differentiable preprocessing

- Added TensorFlow `tf.signal.stft`, mel filterbank matrix multiplication, per-example dB normalization, and a gradient path back to waveform samples.
- Initial TF spectrograms had mean absolute error **0.33554432 dB** and maximum **33.605362 dB** vs librosa; CNN class predictions still agreed 8/8, with max probability difference ~`1.68e-5`.
- Confirmed STFT power under explicit **reflection padding** matched librosa STFT under reflection (relative error ~`1.59e-7`) — but that was not the padding used by our **training-time `librosa.feature.melspectrogram`**.
- Diagnosed mel-power discrepancy near final time frame and found zero-vs-reflection padding mismatch.
- Changed differentiable padding to `tf.pad(..., mode='CONSTANT')` and specified `pad_mode='constant'` in librosa processing. New mean absolute spectrogram error **5.2827595e-6 dB**, max **0.00016021729 dB** on an eight-clip check.
- Verified finite nonzero waveform gradients (example gradient norm ~1.13).

### 5. Optimized first universal waveform

- Implemented untargeted projected sign-gradient **ascent** through frozen CNN and differentiable log-mel: maximize cross-entropy for the original key labels.
- Shared `delta` shape `(11025,)`; `epsilon=0.01`, step size `0.001`, 30 epochs, batch size 32, waveform clipping to `[-1,1]`.
- Evaluated on held-out test recordings using the same UAP added to each waveform (aligned to clip start).
- Observed (488 test examples):

  | Input | Top-1 accuracy |
  |---|---:|
  | Clean | 100.00% |
  | Uniform random waveform, amplitude in `[-0.01,0.01]` | 67.21% |
  | Learned UAP | **3.89%** |

- Clean cross-entropy **0.0038827923**; learned UAP cross-entropy **263.555983**. UAP max amplitude **0.01**, RMS **0.0079888925**.
- **Diagnostic:** all **488** UAP-perturbed recordings predicted as **`b`**, not randomly across 26 classes; characterized as prediction collapse induced by an untargeted optimization.
- Tried generating Gaussian noise at the same RMS as the UAP; its peak reached **0.031383816**, so it **violated** `epsilon=0.01` and is **not an equal-peak control**. Further fairer baselines are needed.
- Added instructions to save mask as `.npy` and optionally 44.1 kHz floating WAV.

### 6. Team handoff docs — 2026-10-17

- Added `PROJECT_OVERVIEW.md` (technical background, preprocessing explanation, current status).
- Added `MODEL_TRAINING_GUIDE.md` (training the authors' factored CNN using shared 26-class contract).
- Added `ADAPTIVE_ATTACKER_GUIDE.md` (adaptive training with **original key labels**, masked-clean mixing, split integrity and evaluations).
- Added this `CHANGELOG.md`.

## Planned — NOT yet completed

- [ ] Train and integrate a stronger 26-way factored CNN; measure accuracy on same-session and transfer-session recordings.
- [ ] Adapt the small CNN to masked/clean examples, preserving original labels; subsequently adapt the stronger CNN.
- [ ] Check prediction confidence for collapsed class `b` and examine confusion matrices.
- [ ] Compare UAP against multiple controls: uniformly random, RMS-and-peak-matched/permuted UAP, phase-randomized controls when suitable, multiple seeds.
- [ ] Epsilon/amplitude sweep, RMS/SNR metrics, clipping fraction and frequency characteristics.
- [ ] Create multiple distinct universal masks, study attacker's recovery as number of masks changes.
- [ ] Add environmental/naturalness penalty and quantify the security–perceptual tradeoff.
- [ ] Evaluate robustness to timing offset, recording session, room, microphone, speaker replay, and mixing realism.
- [ ] Improve packaging/versioning: pinned dependencies, exact train-session provenance, mask metadata manifests, automated smoke tests and CI.

## Reproducibility rules for future changes

1. Every result states **model checkpoint, preprocessing config, class ordering, data split, mask ID, and source-model training split**.
2. Do not silently alter `44100/11025/1024/225/64`, zero-padding, class order `a..z`, or the held-out indices.
3. Track *clean accuracy* and *masked accuracy* for each attacker model; distinguish fixed known masks, unseen masks, white-box optimization and cross-model transfer.
4. Treat 100% clean accuracy and 3.89% UAP accuracy as **within-session digital benchmark** only.
5. Put new changes above the planned section, with an experiment date, a brief rationale, verified metric values and artifact/checkpoint names. Never move planned work into completed without running it.

## References

- <https://github.com/k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack>
- <https://github.com/k3rnel-pan1c-ksd/input-sounds>
