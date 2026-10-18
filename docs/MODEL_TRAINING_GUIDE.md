# Teammate Guide — Train and Integrate the Stronger Acoustic Classifier

**Assignment:** reproduce/adapt the **factored CNN** in the authors' notebook and deliver a stronger **audio-only** attacker model that is interchangeable with our current small CNN. Start by reading `PROJECT_OVERVIEW.md`.

## 1. Where to start

Authors' code: <https://github.com/k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack>  
Dataset: <https://github.com/k3rnel-pan1c-ksd/input-sounds>

The main notebook is **`ZnanstveniRad_final.ipynb`**. Our team may copy this `.ipynb` into our own repository; if so, **you do not need to clone the code repo**. You still need the WAV dataset unless it is separately available in your environment. The authors' README also names `coatnet_vs_cnn_comparison.ipynb`, but the **factored CNN in `ZnanstveniRad_final.ipynb` is the main target**, not the legacy `basic_cnn.keras` checkpoint.

### Local setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow librosa soundfile numpy scikit-learn matplotlib pandas jupyter
# If the dataset is not provided in our repository:
git clone https://github.com/k3rnel-pan1c-ksd/input-sounds.git
jupyter lab
```

### Google Colab setup

```python
!pip -q install librosa soundfile scikit-learn
# Skip this if data is already mounted or shared in Drive:
!git clone https://github.com/k3rnel-pan1c-ksd/input-sounds.git
```

Upload/open the copied notebook. Locate the dataset with `pathlib.Path`; check that `cropped_base_pretraining/a_1.wav` is accessible before training. Notebook paths like `cropped_base/` or `input_sounds_cropped/` in the upstream README may not match the actual downloaded folder names. Do not run the expensive data collection, OCR, or audio-cropping sections when you already have labeled cropped WAVs.

## 2. Read/modify the notebook deliberately

The authors' notebook contains several separate workflows. Focus on **feature extraction, label encoding, `build_cnn` architecture, training, and evaluation**; OCR, click detection, hyperparameter grids, and transfer-learning demos are not required for your first deliverable.

The published factored CNN uses blocks of `3x3`, `3x1`, and `1x3` convolutions with `same` padding, temporal/frequency-aware pooling, batch normalization/dropout and a classification head. The upstream README identifies a selected configuration with convolution filters **(32, 64)**, dense units **256**, dense dropout **0.4**, learning rate **1e-3**. Confirm the exact notebook function rather than transcribing the architecture from memory. Report deviations clearly.

The upstream experiments use a larger key alphabet and different splits. **Do not simply run the full notebook unchanged and assume its output aligns with our UAP code.**

## 3. Non-negotiable team compatibility contract

| Setting | Required |
|---|---|
| Input | `[batch,64,50,1]`, float32 log-mel dB |
| Output | `[batch,26]` with index `0=a ... 25=z` |
| WAV sample rate | 44,100 Hz, mono |
| Clip length | 0.25 seconds / 11,025 samples |
| FFT | 1,024 |
| Hop | 225 |
| Mel filters | 64, librosa default Slaney conventions |
| STFT centering/padding | `center=True`, **`pad_mode='constant'`** |
| dB normalization | `librosa.power_to_db(mel,ref=np.max)`; default `top_db=80` |
| Training dataset | `cropped_base_pretraining`, **a–z only** |
| Labels / split | Reuse `preprocessing_config.json` and `keystroke_dataset.npz` indices |

A different architecture is fine. A different **class order or spectrogram transform is not fine** unless it is deliberately versioned, and all dependent code is updated.

Reference feature extraction:

```python
import librosa, numpy as np

def extract_features(audio, cfg):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=cfg['sample_rate'], n_fft=cfg['n_fft'],
        hop_length=cfg['hop_length'], n_mels=cfg['n_mels'],
        center=True, pad_mode='constant')
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
```

**Important:** do not include SpecAugment in *validation/test preprocessing* or the **waveform-to-feature pathway used during UAP evaluation**. If you use SpecAugment, apply it **only to training data** and record the policy. A spectrogram augmentation is not automatically a physically playable waveform perturbation.

## 4. Recommended training procedure

1. **Load shared arrays and split indices** instead of making new random splits. If the project does not have a committed `.npz`, coordinate to obtain the original split manifest and raw WAVs before running comparisons.
2. Use `labels` integer indices `0..25`. Set the final layer to **26 units**. Prefer `SparseCategoricalCrossentropy` with integer labels and a softmax head; alternatively, use one-hot labels with categorical cross-entropy **consistently**.
3. Extract features from the shared raw waveforms. Shape-check `X.shape == (N,64,50,1)` and inspect a sample visually.
4. Implement the authors' factored CNN, adjusting only its input/output sizes as necessary; keep changes to selected filters/dense parameters documented. Modern Keras is allowed.
5. Train with Adam (`1e-3` as a starting point), batch 32, validation monitoring, and early stopping/checkpointing. Record epochs actually run, seed, validation trajectory, architecture, versions, and dataset counts.
6. Evaluate on `test_idx` **once after model selection**: accuracy, macro-F1, per-class errors/confusion matrix. The original single-session split can give near-perfect scores; treat this as a development benchmark rather than field generalization.
7. When possible evaluate on **`cropped_transfer_learning`** using identical preprocessing and only `a`–`z`; do not silently include its extra classes. This dataset likely represents different conditions. Keep it separate for **out-of-domain testing**, not accidental tuning.
8. Save the final model using Keras' `.keras` format, plus a config manifest and metrics. Make sure loading it with `compile=False` produces the same predictions.

Minimal start of a training notebook:

```python
import json, numpy as np
from tensorflow import keras
with open('preprocessing_config.json') as f:
    cfg = json.load(f)
with np.load('keystroke_dataset.npz', allow_pickle=False) as z:
    waveforms = z['waveforms'].astype(np.float32)
    labels = z['labels'].astype(np.int32)
    train_idx, val_idx, test_idx = z['train_idx'], z['val_idx'], z['test_idx']
assert cfg['classes'] == list('abcdefghijklmnopqrstuvwxyz')
X = np.stack([extract_features(x, cfg) for x in waveforms])[..., None]
assert X.shape[1:] == (64, 50, 1)
# Insert or adapt the author's build_cnn function here.
model = build_cnn(input_shape=(64, 50, 1), n_classes=26)
# Adapt the function signature if the author's build_cnn uses different arguments.
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X[train_idx], labels[train_idx],
          validation_data=(X[val_idx], labels[val_idx]),
          epochs=50, batch_size=32,
          callbacks=[keras.callbacks.EarlyStopping(
              monitor='val_loss', patience=6, restore_best_weights=True)])
model.save('factored_cnn_26.keras')
```

The `build_cnn(...)` line is **integration pseudocode**, not the notebook's guaranteed signature; inspect its definition and adjust it. In particular, the notebook may use one-hot labels: convert the data/loss together, rather than mixing sparse and categorical training.

## 5. Connect it to the existing perturbation framework

The waveform optimizer expects to call:

```python
features = tf_log_mel(waveforms_plus_delta)  # [B,64,50,1]
probs = model(features, training=False)      # [B,26]
```

Thus integration should normally only require replacing the loaded model path:

```python
from tensorflow import keras
model = keras.models.load_model('factored_cnn_26.keras', compile=False)
model.trainable = False
assert model.input_shape[1:] == (64,50,1)
assert model.output_shape[-1] == 26
```

**Important comparability issue:** an existing UAP was trained *against the small CNN*. First test that UAP **without modifying it** on the stronger model (cross-model transfer). Then train a **new UAP specifically on the stronger CNN** using training indices, to measure its own white-box vulnerability. These are distinct experiments; label them accordingly. Avoid evaluating on held-out test clips during iterative UAP tuning.

## 6. Deliverables / definition of done

- `factored_cnn_26.keras` with loadable input/output contract and correct class order.
- `factored_cnn_26_config.json` recording data/feature settings, seed, architecture/training hyperparameters, and version information.
- Notebook or Python training script that runs from clean session after pointing to dataset path.
- Metrics table: base clean test, transfer-session clean performance (when available), confusion matrix and training history.
- One direct interoperability test: inference on shared waveforms with the **existing `tf_log_mel()`**. Differences vs librosa should be tiny; make sure the predicted output index corresponds to the same character as the baseline.
- Brief note specifying whether your reported result is **within-session**, **cross-session**, **cross-model UAP transfer**, or **white-box UAP**.

## 7. Frequent pitfalls

- The old `weights.best.basic_cnn.keras` uses `(64,25,1)`/47 outputs and unknown labels. **Not a replacement.**
- A model with `Dense(67)` or `Dense(47)` cannot be plugged directly into the 26-class UAP pipeline.
- `center=True` with reflection padding is **not** our preprocessing; it caused ~33 dB maximum discrepancies.
- The upstream README describes an `80/10/10` split; our baseline uses `70/15/15`. **Reuse the saved indices** for model-to-model comparisons.
- A 99–100% accuracy random split in homogeneous recordings is not a demonstration of generalization.
- Training your classifier on the small CNN's **test** clips or data chosen after examining test results contaminates comparisons.
- `tf_log_mel()` must remain differentiable when the model is used for gradient-based waveform optimization.

**Handoff contact points:** coordinate filenames/model contract with the UAP owner and give the adaptive-attacker teammate both the clean checkpoint and preprocessing config.
