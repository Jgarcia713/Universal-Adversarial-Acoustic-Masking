# Teammate Guide — Adaptive Attacker Training and Robustness Evaluation

**Assignment:** build a reproducible experiment that tests whether a keystroke classifier can learn to recognize **the original key** despite our universal audio masks. First use the current small CNN, then run the same procedure with the stronger factored CNN when it is available. Read `PROJECT_OVERVIEW.md` first.

## 1. Key concept: the label NEVER changes because of masking

**Question: should masked clips be normal training examples?** **Yes:** train on deliberately perturbed audio while keeping its *original keystroke class label*. This is a standard data augmentation / adversarial training setup for a classifier. Mark examples as masked in **metadata**, not as a new class.

Example:

| Audio given to attacker model | Class to predict | Optional metadata |
|---|---|---|
| Original `a_12.wav` | `a` (index 0) | `clean` |
| `a_12.wav` + universal mask `uap_epsilon_001.npy` | `a` (index 0) | `mask_id=uap_epsilon_001` |
| Original `b_48.wav` | `b` (index 1) | `clean` |
| `b_48.wav` + same universal mask | `b` (index 1) | `mask_id=uap_epsilon_001` |

**Do NOT:** label the universal mask itself as `b` because an earlier CNN collapsed all masked recordings to predicted `b`. That `b` was the model's **wrong prediction**, not the ground truth. Do not introduce class 26 `masked` into the key classification head unless separately implementing a different detection task.

The perturbation file stores **only the mask waveform** (typically 11,025 float32 samples), with metadata about its source model, amplitude bound, sample rate, and training split. The ground truth comes from the accompanying original keystroke recording. Keep optional `mask_id`, `is_masked`, `epsilon`, playback alignment/offset, and source-model ID to facilitate analysis.

## 2. Threat models to distinguish

- **Fixed known mask, attacker aware:** train on audio with the exact known perturbation. This tests how quickly a capable attacker can learn through a static defense, but is optimistic for the attacker.
- **Multiple known masks:** train on a collection of masks and evaluate on held-out recordings with those masks. Test whether diversity requires more data or improves defense.
- **Unseen mask:** train on some masks, then evaluate on separate masks that were **not** used to train the adaptive classifier; tests transfer to new defenses.
- **New recording session / physical replay:** much harder and not yet tested; the initial digital pipeline is only a proof of concept.

Keep these experiment names separate in reports. Accuracy of an **adapted attacker** is the attacker's success (higher = stronger attacker, weaker defense). For defenders, lower attacker accuracy at comparable signal quality is the goal.

## 3. Inputs and compatibility

You need:

```text
keystroke_cnn.keras              # Current small classifier
keystroke_dataset.npz           # waveforms, labels, split indices
preprocessing_config.json       # a-z order and preprocessing
uap_epsilon_001.npy             # universal waveform produced by defense owner
```

The current small model expects `[batch,64,50,1]` log-mel inputs and predicts 26 classes (`a` through `z`). The mask has shape `[11025]` at **44,100 Hz**. Do not resample one without adjusting the other and documenting the operation. Use the **same validated zero-padded librosa transform** or `tf_log_mel`:

```python
import librosa, numpy as np

def extract_features(x, cfg):
    m = librosa.feature.melspectrogram(
        y=x, sr=cfg['sample_rate'], n_fft=cfg['n_fft'],
        hop_length=cfg['hop_length'], n_mels=cfg['n_mels'],
        center=True, pad_mode='constant')
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)
```

Do not use the old checkpoint (`64x25x1`, 47 outputs); its class mapping is unknown. The larger factored CNN must first expose the **same** 26-class API (see `MODEL_TRAINING_GUIDE.md`).

## 4. What to do first: adapt the small CNN

### Step A — keep data partitions separate

- `train_idx`: train/retrain classifier and (for defense owner only) learn UAPs.
- `val_idx`: model selection and early stopping; no fitting weights directly.
- `test_idx`: final evaluation only. **No gradient-based model training, UAP generation, threshold selection, or mask selection on test data.**

**Avoid leakage through masks:** a mask is not independent of its source recordings. The existing UAP was trained using `train_idx`, so treating its waveform as an unseen defense is incorrect. It *is* appropriate for the **known-mask adaptive attacker** experiment, provided the attacker's test is still on `test_idx`. For genuinely unseen-mask experiments, generate distinct masks without tuning their design on final test results, and document which classifier/dataset was used to generate each.

### Step B — mix masked and clean inputs; keep their true labels

For a first run, try a 50% clean / 50% masked probability on each training minibatch. Tune this percentage using validation only; a range like 25%, 50%, 75%, and 100% masked can illuminate the robustness/clean-accuracy tradeoff. Mix a **new mask choice** and (later) jittered placement or strength per sample as the defense expands, rather than relying forever on a single deterministic overlay.

**The first implementation can use saved UAPs as fixed augmentation, with no gradient through the mask.** If optimizing the masker and attacker jointly later, that is a more advanced bilevel/adversarial training task and must be explicitly designed.

Minimal NumPy example illustrating correct label preservation:

```python
import numpy as np
rng = np.random.default_rng(42)

def mix_batch(audio_batch, label_batch, masks, probability=0.5):
    # audio_batch: [B,11025]; labels: [B], unchanged
    x = np.array(audio_batch, dtype=np.float32, copy=True)
    apply = rng.random(len(x)) < probability
    for i in np.flatnonzero(apply):
        mask = masks[int(rng.integers(len(masks)))]
        assert mask.shape == (x.shape[1],)
        x[i] = np.clip(x[i] + mask, -1.0, 1.0)
    return x, np.asarray(label_batch, dtype=np.int32)  # ORIGINAL labels
```

Use only `train_idx` inside the training generator. For validation and final tests, run **separate clean, fixed-mask, and (later) unseen-mask evaluations**, rather than reporting only mixed-condition accuracy.

### Step C — train a fresh/adapted classifier

Start from the clean `keystroke_cnn.keras` weights and continue training on mixed clean/masked data using Adam with a **smaller initial learning rate**, e.g., `1e-4` (tune on validation). Monitor **both validation-clean and validation-masked accuracy**; select a checkpoint using a predeclared criterion (for example, mean of those accuracies or a minimum acceptable clean accuracy).

A clean-to-masked curriculum is reasonable, but it is also worth running a **from-scratch mixed training** baseline as a control. Beware that repeatedly `fit`ting on a single precomputed array containing each training recording duplicated as clean+masked can cause overfitting; online mixing gives more varied exposures.

Pseudocode for the training flow:

```python
# train_idx/val_idx/test_idx remain frozen
# for each epoch:
#   shuffle train indices
#   for each train batch:
#       x_aug, y = mix_batch(waveforms[idx], labels[idx], [saved_delta])
#       features = tf_log_mel(x_aug)  # or verified librosa offline
#       train_step(features, y)       # optimize attacker model weights, NOT mask
#   score CLEAN validation examples
#   score MASKED validation examples
#   checkpoint according to chosen validation rule
# final report: both conditions on test_idx once
```

If masks differ in duration or sample rate, standardize before augmentation rather than silently broadcasting/padding them.

### Step D — evaluate specific questions

Report **top-1 accuracy and macro-F1** for each model/data condition, ideally with confusion matrices:

| Attacker model | Test input | What it measures |
|---|---|---|
| Original clean-trained CNN | Clean | Reference baseline |
| Original clean-trained CNN | Known UAP | Digital masking effectiveness |
| Adapted CNN | Clean | Clean-performance degradation after adaptation |
| Adapted CNN | Known UAP | Attacker recovery when it has seen the mask |
| Adapted CNN | Unseen UAP | Generalization beyond the trained mask |
| Stronger clean-trained CNN | Existing small-CNN UAP | Cross-model transfer |
| Stronger adapted CNN | Known / unseen masks | Adaptive robustness for main attacker |

For fixed-pattern attack, also record **training examples/key**, masking probability, mask count, and attacker training epochs. These are crucial when exploring the project's main question: how attacker accuracy changes with the **number and diversity of universal masking patterns** available during attacker training.

## 5. Suggested staged experiment plan

1. **Sanity check:** load clean small CNN; reproduce ~100% clean and ~3.89% accuracy for the existing UAP under the original 488-example within-session test, including confirmation of prediction collapse to `b`.
2. **Known-mask adaptation:** fine-tune the small CNN on 50/50 clean/masked training batches. Compare clean test accuracy and accuracy on the same UAP, using early stopping on validation only.
3. **Multiple-mask experiments:** request `delta_000.npy`, `delta_001.npy`, etc. with manifest metadata from the UAP owner. Train with 1, 2, 4, ... patterns and plot attacker accuracy against count (controlling number of training examples and total steps where possible).
4. **Mask holdout:** use some masks for adaptive training and others solely for mask-generalization evaluation. Prefer masks generated with independent initializations/constraints or sources, not simply duplicates.
5. **Replace baseline with factored CNN:** repeat the most informative experiments with `factored_cnn_26.keras` once integrated. Compare whether a stronger acoustic classifier recovers more easily.
6. **Realism:** if time permits, test shifted masks, gain changes, noise/reverberation and actual speaker-recording playback. Keep these as separate experiments; digitally mixed samples do not establish physical robustness.

## 6. Storage/metadata proposal

Do **not** export a separate copy of every masked WAV unless it is useful for demonstration or a tool requires on-disk samples. Prefer storing the unmodified `waveforms` dataset once and generating masked training clips on demand. This saves space and allows many-mask experiments.

Suggested folder:

```text
artifacts/
  masks/
    delta_000.npy
    delta_000.wav
    delta_000.json
    delta_001.npy
    delta_001.json
  models/
    keystroke_cnn.keras
    adaptive_small_cnn.keras
    factored_cnn_26.keras
  results/
    adaptive_small_known_mask.csv
    adaptive_small_mask_count.csv
```

Example mask metadata (not a key label):

```json
{
  "mask_id": "delta_000",
  "sample_rate": 44100,
  "n_samples": 11025,
  "epsilon_linf": 0.01,
  "rms": 0.0079888925,
  "source_model": "keystroke_cnn.keras",
  "objective": "untargeted cross_entropy",
  "created_using_split": "train_idx",
  "alignment": "start_of_cropped_0.25s_clip",
  "notes": "digitally mixed, not recorded through speaker"
}
```

A masked-example manifest can contain `original_filename`, `original_label`, `mask_id`, `gain`, `offset_samples`, and `split`. **`original_label` must remain the ground truth**. If a file contains just the perturbation waveform, it has **no key label at all**.

## 7. Implementation details to watch

- **Preprocessing order:** add the waveform perturbation **before** mel conversion; adding masks to log-mel feature arrays is a different, nonphysical operation.
- **Clipping:** the existing UAP evaluation uses `np.clip(x + delta, -1, 1)`. Match that exactly when comparing; clipping itself can generate artifacts, so log clipping statistics as a future control.
- **Data balance:** avoid inadvertently adding more masked `a` than other classes; preserve stratification.
- **Training mode:** use `training=True` for retraining (dropout/BN), `training=False` for validation/evaluation; pretrained model can be fine-tuned, but optimizer state should be documented.
- **Keep original labels:** predictions under mask, including collapsed `b`, do not replace labels.
- **Metrics:** near-100% within-session clean accuracy makes clean test accuracy a weak stress test; cross-session and fresh capture matter.
- **Pattern count study:** if you change number of masks, avoid simultaneously changing dataset size, augmentation probability, and epochs without documenting those confounds.
- **Adversarially generated training inputs vs fixed masks:** our existing delta is learned against a frozen small CNN. If the attacker learns it, the defender would need to create new patterns or optimize against the adapted classifier; coordinate rather than accidentally claim the old UAP is still universal against the new model.

## 8. Deliverables

- Reproducible adaptive training script/notebook that loads the shared config, data, model, and masks.
- `adaptive_small_cnn.keras` and, later, `adaptive_factored_cnn_26.keras` with compatible 26-class outputs.
- Results table (clean, known-mask, unseen-mask when possible) and attacker's learning curves; clear split/mask provenance.
- A mask-count vs attacker-accuracy study, or the framework to support it once multiple masks exist.
- A short written conclusion explaining whether the attacker recovers when retrained and under which masking assumptions.

**For AI assistants:** this is an **adaptive *attacker classifier***, not a classifier of mask IDs. Input is perturbed keystroke audio; output is the original `a`–`z` character. Do not re-label a sample to the CNN's mistaken predicted class. Preserve shared preprocessing, indices, and the model's output mapping.
