# Universal Adversarial Acoustic Masking Against Keystroke Inference

This research project measures how well short masking waveforms reduce a classifier's ability to identify letter keystrokes. It includes a 26-letter convolutional classifier, log-mel preprocessing, the original sign-gradient universal adversarial perturbation (UAP), and a trainer that learns masking residuals across environmental carriers such as rain, fan, and wind.

A **carrier** is an environmental audio recording. A **residual** is a 0.25-second waveform added to the carrier. Paired evaluation compares masking conditions on the same keystrokes, carrier windows, and residual phase offsets. Lower classifier accuracy indicates more effective masking under those evaluation conditions.

## Setup and tests

Use Python 3.10?3.12. Run the commands from the repository root. The experiment configs resolve relative paths from that directory.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
$env:PYTHONHASHSEED = '42'
```

On macOS or Linux, use the following setup instead.

```bash
python -m venv .venv
source .venv/bin/activate
export PYTHONHASHSEED=42
```

In the activated environment, install the package and test dependencies declared in [pyproject.toml](pyproject.toml), then run the tests.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m acoustic_masking --help
```

Run `python -m pytest -q tests/test_preprocessing.py` to check TensorFlow?librosa log-mel equivalence and finite waveform gradients specifically.

## Artifact layout

The [artifacts directory](artifacts/) holds local experiment inputs and generated outputs. Its contents are mostly ignored by Git.

| Folder | Contents |
| --- | --- |
| `artifacts/carriers/` | Original environmental recordings `rain.wav`, `fan.wav`, and `wind.wav`. The loader converts them to mono at 44.1 kHz, applies a 100 Hz high-pass filter, and splits each recording chronologically into 70% train, 15% validation, and 15% test. |
| `artifacts/models/` | The saved classifier `keystroke_cnn.keras`, the prepared dataset `keystroke_dataset.npz` containing waveforms, labels, filenames, and split indices, and preprocessing or training metadata in JSON files. Newly prepared datasets and trained classifiers can also go here under distinct names. |
| `artifacts/masks/` | The historical `residual_bank.npz` and accompanying `manifest.json`, the original `uap_epsilon_001.npy` and its playback WAV, and any individually extracted waveforms such as `rain_phase.npy`. |
| `artifacts/evaluation/` | Paired predictions, accuracy by repeat, summary accuracy, prediction concentration, plots, the evaluation manifest, and the environment record. Separate evaluations can use subfolders such as `scratch_comparison/`. |
| `artifacts/runs/` | One subfolder per residual training run containing `residual.npy`, `history.csv`, `manifest.json`, and `environment.json`. Keep run-specific plots, such as the training curve, alongside the history. |
| `artifacts/results/` | Optional space for manually assembled reports or exports that combine multiple runs. The supplied commands do not write here by default. |

Raw labeled keystroke recordings belong in `input-sounds/cropped_base_pretraining/`. The preparation command reads `a_1.wav` through `z_125.wav` from the original `k3rnel-pan1c-ksd/input-sounds` dataset, skips missing files, and records per-class counts alongside the prepared dataset. Evaluation and residual training use the prepared dataset directly.

## Evaluate the saved baseline

Place the original classifier, prepared dataset, carriers, residual bank, and UAP at the locations above. The supplied [evaluation config](configs/evaluation.yaml) runs 30 repeats per carrier using the nominal keystroke test split and the carrier validation split.

```sh
python -m acoustic_masking evaluate --config configs/evaluation.yaml
python -m acoustic_masking plot-results --evaluation-dir artifacts/evaluation --carrier fan
```

See `accuracy_summary.csv` for the mean and standard deviation across repeats, `accuracy_by_repeat.csv` for repeat-level scores, and `predictions.csv` for individual predictions, carrier starts, and residual phases. To keep a new evaluation separate, use a copy of the config and set its `output_dir` before running it.

## Train and evaluate a scratch residual

The [scratch config](configs/multi_scratch.yaml) trains a zero-initialized residual across rain, fan, and wind for 320 epochs using the saved classifier and training split.

```sh
python -m acoustic_masking train-residual --config configs/multi_scratch.yaml
```

Training saves a new directory under `artifacts/runs/` using the config's `run_name`. Use a distinct name for a repeat run. To compare the retrained waveform with the saved baseline, copy the evaluation config, set a new `output_dir`, and add a separate condition under `residuals` with a `path` pointing to the new run's `residual.npy`. Keep the historical bank entry and the existing evaluation settings for a paired comparison. Run `evaluate` with the copied config and use `history.csv` for the training curve.

Other training configs are [keystroke-only](configs/keystroke_only.yaml), [rain-only](configs/rain_only.yaml), [warm-start multi-carrier](configs/multi_warm.yaml), and [original sign-gradient UAP](configs/original_uap.yaml). Rain-only uses the generalized carrier-aware trainer. For the warm-start run, first extract the saved initial waveform.

```sh
python -m acoustic_masking extract-saved-residual --bank artifacts/masks/residual_bank.npz --key rain_phase --out artifacts/masks/rain_phase.npy
python -m acoustic_masking train-residual --config configs/multi_warm.yaml
```

## Pilot bank for adaptive learning

Train and evaluate the pilot with [mask_bank_pilot.yaml](configs/mask_bank_pilot.yaml).

```sh
python -m acoustic_masking mask-bank train
python -m acoustic_masking mask-bank evaluate-zero
python -m acoustic_masking mask-bank evaluate
```

The pilot runs 180 epochs per mask and compares four seeds under zero and random initialization over eight evaluation repeats per carrier. Each seed uses the same sampling and data order across the two modes. Rerunning training reuses completed runs with matching inputs and settings. The optional `evaluate-zero` stage can run as soon as the four zero-initialized masks are saved. The saved 320-epoch scratch bank entry is the reference for admission, so the comparison to that reference also reflects a difference in training duration. If the previous local retrain is unavailable, omit `reference_run` from the pilot config.

The `evaluation/` subfolder under `artifacts/runs/mask_bank_pilot/` contains the paired predictions, accuracy, and three diversity matrices. The waveform matrix reports maximum absolute circular correlation. The spectral matrix reports Jensen–Shannon distance between normalized power spectra. The behavioral matrix reports the fraction of matched trials where predicted letters differ. The matrices use full gain.

The loudness table reports residual-to-carrier RMS in dB for the complete carrier split and at the fifth percentile of paired carrier-window RMS. It also reports the change in classifier accuracy in percentage points. RMS measures signal level, so audible concealment still requires listening checks. Each admitted mask uses the quietest tested setting that stays within two percentage points of the saved scratch residual on every carrier and passes the diversity screen. `selection.json` records the settings, `candidate_bank.npz` holds all eight raw candidates, and `pilot_bank.npz` stores the admitted waveforms with their selected gain scales applied. A 32-mask expansion is supported when at least six pilot masks pass both screens.

### Compare training checkpoints

The [continuation experiment](configs/mask_bank_continuation.yaml) replays the same eight trajectories from initialization through epoch 320. It captures residual waveforms at epochs 180, 240, and 320 without resetting the optimizer or sampling streams. Run it after the original pilot evaluation.

```sh
python -m acoustic_masking mask-bank train --config configs/mask_bank_continuation.yaml
python -m acoustic_masking mask-bank evaluate --config configs/mask_bank_continuation.yaml
```

Results go to `artifacts/runs/mask_bank_continuation/`. The evaluation checks the epoch-180 replay and compares full-gain accuracy, diversity, and admissions on the same draws. The 240-epoch checkpoint meets the configured efficiency criterion if its classifier accuracy is within one percentage point of epoch 320 or better for every mask and carrier, and it retains every mask admitted at epoch 320.

### Generate the full adaptive-learning candidate bank

The [full bank config](configs/mask_bank_full.yaml) uses the original multi-carrier objective for 320 epochs and produces 16 zero-start and 16 random-start waveforms. It reuses the eight matching 320-epoch runs when available. Each new candidate uses a distinct training seed. Run from the repository root.

```sh
python -m acoustic_masking mask-bank train --config configs/mask_bank_full.yaml --workers 2
```

Outputs go to `artifacts/runs/mask_bank_full_320/`. The runner checks its frozen inputs and settings before reusing a completed run. When both groups finish, it exports `candidate_bank.npz` and `candidate_bank_manifest.json` for adaptive evaluation. The NPZ contains the raw residuals before playback gain. The eight reused waveforms have paired screening results from the continuation experiment.

## Prepare data and train a new classifier

Use these commands if you want to build a fresh dataset and model. The output names keep them separate from the saved baseline inputs.

```sh
python -m acoustic_masking prepare-dataset --audio-dir input-sounds/cropped_base_pretraining --out artifacts/models/keystroke_dataset_new.npz --seed 42
python -m acoustic_masking train-classifier --dataset artifacts/models/keystroke_dataset_new.npz --out artifacts/models/keystroke_cnn_new.keras --seed 42
```

To use these outputs in a residual run or evaluation, set the `dataset` and `model` paths in that run's YAML config.

## Reproducibility and interpretation

- Use the original saved inputs, software versions, and hardware for historical comparisons. Notebook RNG state and historical library versions were only partially recorded. A fresh seeded training run can therefore yield different weights and batch ordering. Save a dependency snapshot with `python -m pip freeze` alongside results that need to be repeated.
- The trainer normalizes each entire carrier split before sampling windows. It uses randomized carrier assignments and circular shifts, clips waveform mixtures, and projects the residual first to the peak budget and then to the RMS budget. Evaluation preserves the historical phase convention and uses `clip_mixtures: false` by default. Set `clip_mixtures: true` for new tests that require bounded mixtures.
- The keystroke split is at the example level. The nominal test split was inspected while developing the masking method, so treat these results as development measurements. Repeats reuse the same keystrokes and vary carrier windows and phases.
- Carrier filtering happens before the chronological split, so small filter effects can cross split boundaries. Independently recorded carriers are needed for source-independent testing. Circular shifts also leave open whether a residual loops seamlessly in playback.
- The package currently supports the fixed classifier and residual bank. Adaptive classifier experiments are described in [docs](docs/) as a separate workflow.

## Source map

| Path | Purpose |
| --- | --- |
| [src/acoustic_masking/](src/acoustic_masking/) | Experiment code. `__main__.py` defines the CLI. `config.py`, `data.py`, `audio.py`, and `preprocessing.py` define settings, load inputs, and build features. `models.py`, `training.py`, and `evaluation.py` implement the experiments. `artifacts.py`, `reproducibility.py`, and `figures.py` handle run outputs, provenance, and plots. |
| [configs/](configs/) | Input paths and experiment settings for training and evaluation. |
| [tests/](tests/) | Regression checks for preprocessing, data splits, audio operations, and paired evaluation. |
| [notebooks/](notebooks/) | Earlier notebook experiments. |
| [docs/](docs/) | Project background and supporting research guides. |
