# Teammate 2 — Adaptive Attacker: K-Mask Recovery Experiment

**Priority: P0; this is the central remaining experiment.** Train an attacker to predict the **original keystroke letter** from masked audio and measure recovery as the number K of known training masks increases. Deliver a verified K-versus-accuracy curve and per-condition predictions, **not** another mask generator.

Read `00_PROJECT_STATUS_AND_INTEGRATION.md` first. Inspect the checked-out repo's latest mask-bank manifest and audio-mixing/evaluation code before using any examples below. The bank was reportedly built from 32 candidates; do **not** assume all 32 pass admission or that the current bank already has a stable `MaskBank` class.

## 1. Define the threat model in the report

**Known-mask adaptive attacker:** has examples digitally mixed with a known selection of K defender masks. The original key label remains the supervised target. The attacker may know the defense mechanism and mask IDs/values for its training subset, but must not train on other mask IDs reserved for evaluation.

**Unseen-mask evaluation:** test the adapted model on new keystroke examples mixed using mask waveforms **not available during that model's training, tuning, or stopping**. This is not the same as an independent unseen microphone or carrier recording.

**K = 0 control:** train with **carrier-only** mixtures under the same carrier sampling and training budget. Compare to an unadapted saved CNN (clean-trained model) as a separate reference, not as K=0 unless its training distribution matches the control.

Accuracy = correct original letters / number of test examples. Higher adaptive accuracy means the attacker recovers more; lower means the defense remains more effective. A large K may help an attacker generalize or make fixed-budget training harder — **do not assume the expected direction**.

## 2. Required inputs and parsing

1. Prepared `artifacts/models/keystroke_dataset.npz`: call `acoustic_masking.data.load_dataset` and use `waveforms`, `labels`, and **the actual indices** `train_idx`, `val_idx`, `test_idx`. Do not infer a new random split or treat the NPZ as a masked-example bank.
2. Mask bank: inspect bank manifest (format version, IDs, `.npy` paths, seed, initialization, qualifications, gain, hashes). Bank might be 32 individual run directories rather than a single NPZ. If historical `residual_bank.npz` is used, inspect `.files` and extract waveforms by exact documented keys; this is **not** interchangeable with the dataset NPZ.
3. Environmental recordings: use the repo loader and the documented train/val/test carrier splits. Reuse the repo's **per-carrier target RMS** and **residual gain**, not a new amplitude heuristic.
4. Model: first adapt the saved small CNN for minimal integration risk; repeat with the second model if delivered in time, using a **separate run family**.

```python
from acoustic_masking.data import load_dataset
import numpy as np

d = load_dataset('artifacts/models/keystroke_dataset.npz')
x_train, y_train = d['waveforms'][d['train_idx']], d['labels'][d['train_idx']]
x_val, y_val = d['waveforms'][d['val_idx']], d['labels'][d['val_idx']]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# For a historical bank only, inspect keys before accessing:
with np.load('artifacts/masks/residual_bank.npz', allow_pickle=False) as bank:
    print(bank.files)  # NOT waveforms/labels/train_idx!
```

**Critical shape assertions:** each keystroke and each mask is 11,025 samples at 44,100 Hz. Output is 26 classes with `a=0..z=25`, features `[B,64,50,1]`. Reuse `src/acoustic_masking/preprocessing.py` rather than writing a new STFT. Run `pytest` before writing training code.

## 3. Mixing protocol — must match the defense

For each permitted keystroke `(x, y)`:

1. Randomly choose one of rain/fan/wind (or use the **exact same** documented carrier sampling distribution for every K).
2. Sample carrier from the **training source split for training**, validation carrier split for validation, and designated evaluation split for evaluation. Avoid letting identical source segments leak across stages.
3. Match the repo's carrier normalization: normalize **whole split**, sample windows, apply `carrier_rms`; do not normalize each window independently. Reuse `carrier_config` (historical rain target RMS 0.005, fan/wind 0.010; residual gains may differ; verify in current configs).
4. For K>0, uniformly sample one of the K **training-known mask IDs**, apply random **circular** phase, and multiply by the same `playback_scale` and `residual_gain` as the comparison settings. For K=0, omit only the residual, **not** the carrier.
5. Apply **one explicit and shared clipping convention** across K and evaluation. Historical defense paired evaluation used `clip_mixtures: false`, training clipped. The new adaptive study should document and standardize the policy; do not quietly switch by condition. Use existing mixing helpers or refactor them into a single tested function if necessary.
6. Convert waveform batch to log-mel using the validated repo transform. Supervise with the **original** y; never label an example as the wrong `b` prediction or create a `masked` class.

Store `sample_id`, carrier source ID, carrier-window start, phase offset, mask ID and gain in per-example metadata for debugging/replay. Example-level mixing should be stochastic in *training* but paired/fixed in *evaluation*.

## 4. Freeze a bank split BEFORE adaptation

**Proposal only — finalize according to the actual bank manifest:** select **24 training-eligible masks + 8 unseen masks** from 32 verified candidates, balanced by random/zero initialization and effectiveness. If fewer qualify, use a smaller split and adapt K values to it. Save IDs and mask waveform hashes in a **versioned bank split manifest**. Do not choose subsets based on adaptive results.

To compare K:

- Use **nested** subsets inside each subset-replicate: e.g., K1's mask is included in K2; K2 included in K4; and so on. Store their exact ordered IDs.
- Use multiple *independent subset-replicates* if time permits (e.g., 3), since an unusually weak/strong mask can dominate results.
- A starting grid is **K = 0, 1, 4, 8, 16**; add 2 and 24 only if time. Do not propose K=32 with 8 held-out masks in a 32-candidate bank.
- When evaluating each model, report both **seen-mask** performance (masks in its K subset) and **unseen-mask** performance (the same fixed reserve across all K).
- For a fair fixed-budget comparison, hold the model initialization policy, optimizer, number of **updates**, batch size, carrier sampling distribution and *total examples* constant as K changes. With K increasing, per-mask exposure decreases; **this is part of the fixed-budget question**. Do not accidentally increase epochs/steps per K.
- Optional secondary result: updates/examples needed to recover to a chosen accuracy threshold. Report this separately from the fixed-budget curve.

## 5. A clear minimal experimental matrix

| Experiment | Training masks | Evaluation groups | Purpose |
| --- | --- | --- | --- |
| Unadapted CNN | None (frozen) | Clean, carrier-only, known and reserved masks | Original defense effect |
| K=0 | Carrier only | Carrier-only, fixed reserved masks | Carrier adaptation baseline |
| K=1 | 1 bank mask | Same 1 mask; reserved 8 | Static-mask recovery / transfer |
| K=4 | 4 masks | Same 4; reserved 8 | Limited diversity adaptation |
| K=8 | 8 masks | Same 8; reserved 8 | Intermediate diversity |
| K=16 | 16 masks | Same 16; reserved 8 | Larger diversity |

All evaluations should use **identical keystroke IDs and replayable carrier windows/phase draws** across compared K models, with the same gain, clipping, model evaluation mode, and preprocessing. Clean and carrier-only accuracy must be tracked as sanity checks; adaptation could degrade them. For K>0, seen-mask averages need carefully chosen weighting: uniform over mask IDs and carrier conditions; save per-mask/per-carrier results to expose variation.

**Data warning:** the nominal `test_idx` is a development set because the defender already examined it. Use it to finish a controlled class project, but label it accurately. If the defense owner obtains independent new-session recordings, run a separate **independent test** without using them to tune K or mask selection.

## 6. Implementation milestones (time-boxed)

### P0-A. Loader + mixture parity (first)

- Build/read an efficient mask-bank loader with explicit chosen mask IDs and validation of waveform shapes, dtypes and file hashes.
- Reuse shared carrier and transform modules. Write a test proving **K=0** equals the carrier-only mixing pipeline on identical draws, and that **K=1** produces exactly the expected phase-shifted residual mixture.
- Run one batch end-to-end through the small CNN. Check labels unchanged, finite waveform and spectrogram, no clipping surprise.

### P0-B. One-mask sanity experiment

- Fine-tune **a copy** of the saved CNN or train a fresh same-architecture CNN; choose one protocol and use it consistently for K=0,1,4,... . Record whether original weights are reused (initialization must match across K).
- Train a short K=1 run (small number of steps) and verify loss decreases on masked training examples. Evaluate on held-out keystroke IDs and the *same* known mask.
- Compare K=1 to frozen unadapted and K=0. If no improvement, debug mixing, label mapping, gradient flow and optimizer before scaling.

### P0-C. K sweep and outputs

- Execute K=0,1,4,8,16 using the same training budget and deterministic subset manifests. Keep a checkpoint chosen by a **fixed rule** (e.g., last update; or validation selection applied consistently); reserve unseen masks from selection.
- Save per-example predictions, confusion matrices, per-mask/per-carrier accuracy and aggregate results. Ideally run ≥2–3 training seeds/subset replicates, but **one completely validated sweep is better than many inconsistent runs**.
- Generate a plot of attacker accuracy vs K, with separate lines for seen and unseen masks and (where sufficient replicates exist) variability/error bars. Report raw sample sizes.

### P1-D. Stronger-model replication

- If new compatible model is available and stable, repeat the key K cases (at least 0,1,4,16) under one documented budget. Do **not** secretly fine-tune the new model with an altered preprocessing or mask selection scheme.

## 7. Requested implementation artifacts

Adapt paths to current repo. Suggested (not yet implemented):

```text
configs/adaptive_attack.yaml
src/acoustic_masking/adaptive.py          # model train/eval loop
src/acoustic_masking/mask_bank.py         # ONLY if not already present; prefer existing loader
scripts/ or CLI subcommand              # train-adaptive / evaluate-adaptive
artifacts/results/adaptive/
  bank_split_manifest.json
  k_subsets.json
  run_manifest.json
  training_history.csv
  evaluation_predictions.csv
  accuracy_by_k_carrier.csv
  accuracy_by_k_mask.csv
  summary.csv
  accuracy_vs_k.png
```

**Proposed minimum columns** for prediction CSV: `run_id, model_id, seed, K, subset_id, split, source_sample_id, true_label, pred_label, carrier, carrier_window_start, mask_id, seen_or_unseen, residual_gain, playback_scale, clip_mixtures, correct`. For summary: `K, subset_id, training_seed, split, evaluation_group, carrier, n_examples, n_correct, accuracy`. Ensure no artificial inflation of N from multiple phases; document repeated measures.

## 8. Required tests and pass/fail checklist

- [ ] `python -m pytest -q` passes before/after work.
- [ ] Assert feature `[B,64,50,1]`, classifier `[B,26]`, masks `[11025]`, samples `[B,11025]`.
- [ ] Assert masked labels equal original labels; all K conditions use identical keystroke train/val/eval partitions.
- [ ] Assert each held-out mask ID is absent from *all* K training subsets, checkpoint decisions and tuning.
- [ ] Verify K=0 numerically matches carrier-only; K=1 matches direct reference mixture at fixed carrier window and phase.
- [ ] Check noise/carrier levels and residual gains against existing config; record clipping choice.
- [ ] Verify fixed budget: same update count and number of training examples for every K.
- [ ] Save one CSV with raw predictions, one with aggregations, and one figure-ready K curve; link exact commands and paths in PR.

## 9. Coding-agent brief (copy into your agent)

> Implement a bounded adaptive-attacker experiment in the existing acoustic-mask repo. Read the main README, `docs/00_PROJECT_STATUS_AND_INTEGRATION.md`, this guide, and actual `data.py`, `audio.py`, `preprocessing.py`, `evaluation.py`, `training.py`, and bank-manifest structure before changing anything. Use the prepared keystroke NPZ via the official loader; preserve 0=a..25=z labels and the `[B,64,50,1]` log-mel transform. Reuse carrier-window normalization, gains, and phase-shift conventions. Freeze a mask-bank 24/8 train/unseen split (only if enough masks qualify), and implement fixed-budget K=0,1,4,8,16 runs with equal update counts. Evaluate on paired examples with seen and held-out masks, save predictions/config/seeds/checkpoints and produce a K-versus-accuracy plot. Add tests for format, label and split correctness, K=0/K=1 mixture parity and held-out mask exclusion. Never overwrite existing bank/masks or claim nominal test recordings are an independent holdout. Prioritize a working, reproducible sweep over extra features. Return a PR with run commands and methodological caveats.

## 10. Do not spend the remaining weeks on

- Building a brand-new end-to-end acoustic dataset format when `.npz` already works.
- Retraining or selecting defensive masks based on adaptive results.
- Changing the defense optimization objective, gain per K, preprocessing, or label space.
- Calling distinct phase offsets independent test keystrokes.
- Physical microphone evaluation, unseen keyboard deployments, new adversarial attacks and large hyperparameter searches. These are future work unless core K study is completed and written.
