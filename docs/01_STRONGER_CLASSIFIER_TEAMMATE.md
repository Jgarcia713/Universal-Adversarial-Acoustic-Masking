# Teammate 1 — Stronger Acoustic Keystroke Classifier

**Priority: P0.** Deliver an independent, stronger **26-letter audio classifier** that can be evaluated through the shared pipeline. **Do not get stuck reproducing a published notebook in full.** The objective is a defensible second model and a clean model-generalization comparison by the deadline.

Read `00_PROJECT_STATUS_AND_INTEGRATION.md` first, then inspect the main branch. In particular, open `src/acoustic_masking/data.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `configs/evaluation.yaml`, and `tests/test_preprocessing.py`.

## 1. Your scientific question

Does adversarial masking that was optimized against the original small CNN also affect a different / stronger attack model? You are delivering a model plus **reproducible evidence**, not merely a checkpoint. The baseline small CNN's ~100% clean accuracy on its original random-file split does not mean a more complex model must beat it on that split; a stronger model can be justified by more realistic generalization, architectural independence, or robustness. Do not claim 'stronger' solely because parameter count is larger.

## 2. Input and model interface (mandatory)

- **Output:** one probability/logit score per class `[B,26]`, fixed order `a..z` (`0..25`). Record whether outputs are logits or probabilities; use the corresponding loss.
- **Input preprocessing:** `librosa_log_mel` / `LogMel` from the current repo only, `[B,64,50,1]`, float32 (44.1kHz, 11,025-sample mono waveform, FFT 1024, hop 225, zero padding, 64 mel bins, dB relative to max).
- **Data source:** `artifacts/models/keystroke_dataset.npz` loaded with `acoustic_masking.data.load_dataset`. Required keys `waveforms`, `labels`, `train_idx`, `val_idx`, `test_idx`. Do not substitute another split without assigning it a different dataset ID.
- **Frozen original CNN:** `artifacts/models/keystroke_cnn.keras` must remain untouched; output your model under a new filename and record training history/config.
- **Raw WAVs:** `input-sounds/cropped_base_pretraining/` uses `[letter]_[1..125].wav`; do not include nonletter classes unless proposing a separate, explicitly incompatible experiment.

Sanity check:

```python
from acoustic_masking.data import load_dataset
import numpy as np

d = load_dataset('artifacts/models/keystroke_dataset.npz')
print({k: d[k].shape for k in ('waveforms','labels','train_idx','val_idx','test_idx')})
assert d['waveforms'].shape[1] == 11025
assert np.all((0 <= d['labels']) & (d['labels'] < 26))
```

**Model shape test:** run an arbitrary **unmasked** batch of `[B,11025]` through `LogMel` and the saved/new models; verify preprocessing shape `[B,64,50,1]` and output `[B,26]`. Run the repo tests before proceeding.

## 3. Implementation plan

### P0-A. Establish the baseline before modifications

1. Run `python -m pytest -q` and `python -m pytest -q tests/test_preprocessing.py`.
2. Run the existing saved-CNN evaluation on the current config, writing to a **new** `output_dir` (copy its YAML; do not overwrite baseline outputs).
3. Record clean accuracy, per-letter confusion, model/dataset hash and exact test examples. The 'test' split has been previously used for development; call it that in the report.
4. Confirm labels and spectrogram features by printing a few examples from known class IDs. Confirm no model input accidentally receives `uint8`, raw audio where spectrogram expected, or misordered axes.

### P0-B. Implement one alternative architecture

Start with the **factored CNN** from the upstream `ZnanstveniRad_final.ipynb` / prior teammate guide if it is available. The reference architecture uses factorized `3x1` and `1x3` kernels as well as `3x3` convolutions. **Inspect its actual code before implementing and cite its source.** If obtaining/reproducing it would block progress, implement one clearly described conventional CNN that differs meaningfully from the frozen baseline. Do **not** silently label a custom model 'published factored CNN'.

- Keep the same preprocessing and class mapping to permit a direct model comparison.
- Train using `train_idx`; tune hyperparameters and early stopping with `val_idx` only.
- Set and record all seeds, optimizer, batch size, epochs, checkpoint rule, sample weights, software versions.
- Save full model, label mapping, training history, confusion matrix and metadata.
- Keep architecture changes *modest*: one credible model is more valuable than a huge search with no final test.

### P0-C. Validate on two conditions

1. **Clean key classification:** report baseline small CNN vs new classifier on identical example IDs and splits. Include per-letter confusion and, if possible, a paired difference with uncertainty. If both are near perfect, state that discrimination is limited by the easy original dataset.
2. **Transferred mask evaluation:** freeze the new classifier and evaluate historical carrier-only, random-noise, and the bank masks **without optimizing masks against the new classifier**. This distinguishes transfer to an unseen model from white-box masking. Use identical carrier windows, phases, gain and clipping in paired conditions. Report separate fan/rain/wind accuracies, not only an overall average.

If time allows, use a truly independent recording session / keyboard / microphone; make sure it is labelled with the same 26 letters and normalized to the same sample rate and clip duration. This is the strongest way to evaluate whether new-model clean accuracy is meaningful. **Do not mix an entirely new recording condition into the old random split and claim it is independent.**

### P1. Optional: separate generalization analysis

A new dataset or session is useful **only if** class labels, acquisition setup, microphone, keyboard, and any necessary segmentation process are documented. Report whether the model was retrained/fine-tuned on the new recordings. A result after fine-tuning is **not** zero-shot generalization.

## 4. Required outputs

Proposed output structure — adapt to the actual project CLI, do not create duplicate wrappers unnecessarily:

```text
artifacts/models/<new_model_name>.keras
artifacts/results/stronger_model/
    config.yaml
    model_manifest.json
    history.csv
    clean_predictions.csv
    confusion_matrix.csv
    mask_transfer_by_carrier.csv
    notes.md
```

CSV fields for mask-transfer study: `model_id, carrier, condition, mask_id, repeat, n_examples, n_correct, accuracy, seed, gain, clip_mixtures`. Include original filename/example index when writing per-example predictions. Save confusion plots if useful for the report. **These filenames/columns are proposed handoff standards, not a statement that the current CLI already writes them.**

## 5. Tests and acceptance criteria

- [ ] Runs `python -m pytest -q` and log-mel equivalence test without breaking existing modules.
- [ ] A fixed set of example waveforms produces the same feature tensors used by the baseline evaluation; model returns `[B,26]` and the correct label indices.
- [ ] Full training runnable from one documented command; checkpoint + configuration + history committed or accessible as artifacts (large binaries may be ignored by Git; provide shareable location and checksums).
- [ ] No training/evaluation split overlap, and no preprocessing drift.
- [ ] Reproduce at least one clean and one masked comparison from saved artifacts.
- [ ] Submit a PR containing implementation, tests, a short result table, caveats and a ready-to-use figure.

## 6. Coding-agent brief (copy into your agent)

> You are implementing the second 26-letter acoustic-keystroke attacker for a time-bounded research project. Read the repo README, `docs/00_PROJECT_STATUS_AND_INTEGRATION.md`, the current data/preprocessing/model/evaluation modules, and existing tests first. Preserve the exact `[B,64,50,1]` log-mel feature convention and `0=a..25=z` labels; use the `keystroke_dataset.npz` split indices and do not train on evaluation data. Implement one credible alternative architecture (prefer factored CNN if its actual source is available), save model and reproducibility information, add compatibility tests, and compare clean and masked accuracy against the saved small CNN using paired examples. Do not modify the original classifier, bank or historical configs in place. Return a PR, runnable commands, summary results and limitations; ask only if a necessary artifact is truly missing.

## 7. Stop conditions / future work

If the factored model is blocked by unavailable upstream weights/notebook code, a reproducible alternative 26-class CNN + transfer evaluation is an acceptable minimum. Stop investigating OCR, full 67-key class spaces, architecture sweeps, and physical microphone deployment until final results/report exist. An honest failure to generalize is itself reportable; don't hide it.
