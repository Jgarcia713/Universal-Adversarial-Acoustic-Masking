# Stronger classifier: factored CNN and mask transfer

The question: do the masks, all optimised against the original small CNN, also hurt a different and stronger keystroke classifier that never saw them?

**Answer: partly.** The masks transfer fully on rain, about three-quarters on wind and about a third on fan. The new model starts out more accurate under background noise and stays more accurate under masking, so it remains the stronger attacker.

## Getting the model

The model file is 158 MB, which is over GitHub's 100 MB limit, so it is not in Git.

- Download: https://drive.google.com/file/d/1J1M8Gr1_fUtWmpMIRabZAvv1puJN3T6v/view?usp=sharing
- Put it at `artifacts/models/factored_cnn.keras`. Every config and results file refers to that path.
- Check the download: sha256 `b23b1b4abe2183fc8c0d21157ae269afe04b24241963dc852b285ab587f64bd3`.

## The model

- **Source.** This is the factored CNN from `notebooks/ZnanstveniRad_final.ipynb`, cell 14, using that notebook's final "paper config 13" hyperparameters from cell 19. It is implemented as `build_factored_cnn` in `src/acoustic_masking/models.py`.
  - Changes from the notebook: 26 letter classes, integer labels instead of one-hot, and this repo's log-mel features.
- **Architecture.** There are two convolution blocks with 32 and then 64 filters. Each block applies a 3×3, a 3×1 and a 1×3 convolution, then 2×1 max pooling. The pooling halves the mel (frequency) axis and keeps all 50 time frames. A dense layer of 256 with batch norm follows, then a softmax output.
  - 13,164,794 parameters, against 490,778 in the original CNN.
- **How it differs from the original CNN.** The original has three blocks of 3×3 kernels only, with no padding, 2×2 pooling over both axes and a dense layer of 128.
- **Interface.**
  - Input: `[B, 64, 50, 1]` from `LogMel` or `librosa_log_mel`.
  - Output: `[B, 26]` probabilities (softmax), with a..z = 0..25.
  - Loss: sparse categorical cross-entropy.
- **Training data.** The network trained on `train_idx` (2,275 clips) only. `val_idx` (487 clips) was used only for early stopping. `test_idx` (488 clips) played no part in training or model selection. The clips were clean keystrokes, with no augmentation and no sample weights.
- **Training settings.**
  - Adam with learning rate 1e-3, batch 32, up to 50 epochs.
  - Early stopping on validation loss with patience 5, restoring the best weights. Training stopped after 34 epochs. The best epoch was 29, with validation loss 0.0011.
  - Seed 42, `PYTHONHASHSEED=42` and TensorFlow op determinism.
  - Each epoch took about 20 s on a 4-core laptop CPU.
- **Training record.** The full settings are in `config.yaml`. The history is in `artifacts/models/factored_cnn_history.csv`.

## Results

### Clean keystrokes (no carrier, no mask)

| Model | Correct (test split) | Lowest confidence |
|---|---|---|
| Original CNN | 488 / 488 | 0.708 |
| Factored CNN | 488 / 488 | 0.969 |

Neither model got any clip wrong that the other got right (exact McNemar p = 1.0). Both are perfect, so clean accuracy on this dataset cannot rank them. That points to an easy, single-session dataset, not to real-world accuracy.

### Masked keystrokes (transfer)

**Setup.**
- **Configs.** `configs/transfer_cnn.yaml` and `configs/transfer_factored.yaml` differ only in the model and the output folder. Each matches `configs/evaluation.yaml` except that it adds the 32 candidate-bank masks.
- **Evaluation settings.**
  - Seed 20261116, 30 repeats, the 488 test keystrokes, and carrier windows drawn from each carrier's validation split.
  - Playback scale 0.9. Residual gain 0.75 for fan and 1.0 for rain and wind. `clip_mixtures: false`.
- **Pairing.** Both runs were checked to use identical keystrokes, carrier windows and mask phases.
- **No exposure.** The factored model never saw the masks during training or mask optimisation.

Accuracy (%), mean of 30 repeats:

| Condition | Fan: original | Fan: factored | Rain: original | Rain: factored | Wind: original | Wind: factored |
|---|---|---|---|---|---|---|
| Carrier only | 56.75 | 62.38 | 59.11 | 75.68 | 71.30 | 86.56 |
| Random noise | 53.18 | 62.51 | 56.78 | 73.65 | 67.09 | 84.36 |
| Original UAP | 53.89 | 62.61 | 55.83 | 73.42 | 66.92 | 83.46 |
| Multi-carrier (scratch) | 44.16 | 58.28 | 49.58 | 65.83 | 54.02 | 74.23 |
| Bank mean (32 masks) | 44.14 | 57.96 | 49.59 | 65.65 | 53.68 | 73.78 |

Accuracy drop versus carrier only, in percentage points. The transfer ratio is the factored drop divided by the original drop:

| Condition | Fan: original / factored / ratio | Rain: original / factored / ratio | Wind: original / factored / ratio |
|---|---|---|---|
| Random noise | 3.6 / −0.1 / — | 2.3 / 2.0 / — | 4.2 / 2.2 / — |
| Original UAP | 2.9 / −0.2 / −0.08 | 3.3 / 2.3 / 0.69 | 4.4 / 3.1 / 0.71 |
| Multi-carrier (scratch) | 12.6 / 4.1 / 0.33 | 9.5 / 9.9 / 1.03 | 17.3 / 12.3 / 0.71 |
| Bank mean (32 masks) | 12.6 / 4.4 / 0.35 | 9.5 / 10.0 / 1.05 | 17.6 / 12.8 / 0.73 |

**What the results show.**
- **The mask effect is consistent.** The multi-carrier mask lowered accuracy in all 30 repeats, for both models, on every carrier. The drop varies by about 1 point (std) across repeats.
- **All 32 bank masks behave alike.**
  - Original CNN, per-mask drops: fan 12.4–12.9, rain 9.3–9.8, wind 17.2–18.0.
  - Factored CNN, per-mask drops: fan 4.1–4.8, rain 9.5–10.6, wind 12.0–13.5.
- **The masks work beyond extra noise alone.** The random-noise control, played at the same gain, barely affects the factored model, at most 2.2 points. The original UAP has almost no effect on fan for either model.
- **The factored model stays the stronger attacker.** Under the bank masks it still reads 58% of keystrokes on fan, 66% on rain and 74% on wind. The original CNN reads 44%, 50% and 54%.
- **Hypothesis, untested:** transfer may follow how loud the mask is relative to the carrier. That ratio is highest for rain (gain 0.9, carrier RMS 0.005) and lowest for fan (gain 0.675, carrier RMS 0.010), which matches the rain > wind > fan ordering. With three carriers this is only a suggestion; the fixed-gain sweep could test it.

Figure: `transfer/transfer_by_carrier.png`.

### Reproduction check

Running the original CNN through `configs/transfer_cnn.yaml` reproduced the saved multi-carrier (scratch) accuracies in `docs/00_PROJECT_STATUS_AND_INTEGRATION.md` exactly: fan 44.16%, rain 49.58% and wind 54.02%.

## Limitations

- **Development evaluation, not a final test.** The 488 `test_idx` keystrokes were also used while the masks were developed.
- **Repeats are repeated measures.** Every repeat reuses the same 488 keystrokes; only the carrier windows and mask phases change. The spread across repeats therefore understates the uncertainty for new keystrokes.
- **One training seed.** The factored model was trained only once.
- **Digital mixing.** These results come from digital mixing with recorded carriers on a fixed 26-letter dataset. Nothing was played through a speaker or tested over the air.
- **No independent-session test.** Both models score 100% on clean clips, and no recordings from another keyboard or session were tested.
- **Short fan carrier.** The fan carrier's validation split is short, about 1.4 s, so fan windows overlap heavily.
- **Two machines.** The factored masked evaluation ran on a 16-core Linux cloud machine with AMD CPUs. Everything else ran on a Windows laptop with an Intel CPU. Library versions were the same and the random draws were verified identical, but tiny floating-point differences between machines cannot be ruled out.
- **Requirements file conflict.** `requirements.txt` (UTF-16, which pins TensorFlow 2.21 and librosa 1.0) disagrees with `pyproject.toml`. These runs used Python 3.12.10, TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.5.5 and librosa 0.11.0, on CPU.

## Reproduce

Run from the repo root with `PYTHONHASHSEED=42` set.

```bash
# 1. Train (about 12 min on a 4-core laptop CPU)
python -m acoustic_masking train-classifier --arch factored --dataset artifacts/models/keystroke_dataset.npz --out artifacts/models/factored_cnn.keras --seed 42 --epochs 50

# 2. Clean comparison (under a minute)
python -m acoustic_masking evaluate-clean --model cnn=artifacts/models/keystroke_cnn.keras --model factored=artifacts/models/factored_cnn.keras --split test --out artifacts/results/stronger_model/clean

# 3. Masked evaluation, one run per model (about 1 h for the CNN on a 4-core laptop; the factored model took 25 min on 16 cores)
python -m acoustic_masking evaluate --config configs/transfer_cnn.yaml
python -m acoustic_masking evaluate --config configs/transfer_factored.yaml

# 4. Combine into the transfer tables and figure (about a minute)
python -m acoustic_masking summarize-transfer --evaluation cnn=artifacts/evaluation/transfer_cnn --evaluation factored=artifacts/evaluation/transfer_factored --out artifacts/results/stronger_model/transfer
```

Training again from scratch may not give a bit-identical model on different hardware. To reproduce these exact numbers, use the Drive model and skip step 1.

## Files

| What | Where |
|---|---|
| Model | `artifacts/models/factored_cnn.keras` (Drive) |
| Training settings | `config.yaml` |
| Checksums, label order, interface | `model_manifest.json` |
| Training history and metrics | `artifacts/models/factored_cnn_history.csv`, `factored_cnn.json`, `factored_cnn_environment.json` |
| Clean predictions, both models | `clean/clean_predictions.csv` |
| Confusion matrices (rows = true letter) | `clean/confusion_cnn.csv`, `clean/confusion_factored.csv` |
| Clean accuracy and paired test | `clean/clean_summary.csv`, `clean/paired_comparison.csv` |
| Mask transfer, one row per model × carrier × condition × repeat | `transfer/mask_transfer_by_carrier.csv` |
| Averages and side-by-side comparison | `transfer/transfer_summary.csv`, `transfer/transfer_comparison.csv` |
| Figure | `transfer/transfer_by_carrier.png` |

These files are not in Git:
- **Per-prediction outputs.** `artifacts/evaluation/transfer_cnn/predictions.csv` (77 MB) and `transfer_factored/predictions.csv` (76 MB). The repo's `.gitignore` excludes `artifacts/evaluation`. Step 3 regenerates them, and their checksums are in `model_manifest.json`.
- **The training checkpoint.** `artifacts/models/factored_cnn_best.keras` has weights bitwise-identical to `factored_cnn.keras`, so it is not shared.
