"""Command-line entry point for the acoustic masking experiments.

Dispatches commands to prepare data, train the classifier or a masking
residual, evaluate saved waveforms, plot results, and extract a bank entry.
Residual training and evaluation read input paths and settings from YAML.
"""
import argparse
import json
from pathlib import Path


def prepare_dataset(args):
    import numpy as np
    from acoustic_masking.data import load_letters,split_indices
    from acoustic_masking.config import CLASSES,SAMPLE_RATE,N_SAMPLES,N_FFT,HOP_LENGTH,N_MELS
    from acoustic_masking.reproducibility import sha256

    waveforms, labels, filenames = load_letters(args.audio_dir)
    splits = split_indices(labels, seed=args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, waveforms=waveforms, labels=labels, filenames=filenames, **splits)
    info = {
        'classes': CLASSES,
        'sample_rate': SAMPLE_RATE,
        'n_samples': N_SAMPLES,
        'n_fft': N_FFT,
        'hop_length': HOP_LENGTH,
        'n_mels': N_MELS,
        'pad_mode': 'constant',
        'db_reference': 'per sample maximum',
        'seed': args.seed,
        'per_class_counts': {letter: int((labels == i).sum()) for i, letter in enumerate(CLASSES)},
        'dataset_sha256': sha256(path),
        'warning': 'Example-level split only; not recording-session independent',
    }
    path.with_suffix('.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Wrote {len(labels)} examples to {path}')


def train_classifier(args):
    import numpy as np
    from tensorflow import keras
    from acoustic_masking.data import load_dataset
    from acoustic_masking.preprocessing import librosa_log_mel
    from acoustic_masking.models import build_cnn
    from acoustic_masking.reproducibility import set_seed,save_environment,sha256

    set_seed(args.seed)
    data = load_dataset(args.dataset)
    features = np.stack([librosa_log_mel(wave) for wave in data['waveforms']])[..., None]
    labels = data['labels']
    train, val, test = (data[key] for key in ('train_idx', 'val_idx', 'test_idx'))
    model = build_cnn()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(destination.with_name(f'{destination.stem}_best.keras')),
            monitor='val_loss', save_best_only=True,
        ),
    ]
    model.fit(
        features[train], labels[train], validation_data=(features[val], labels[val]),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1,
    )
    model.save(destination)
    loss, accuracy = model.evaluate(features[test], labels[test], verbose=0)
    metrics = {
        'seed': args.seed,
        'test_accuracy': float(accuracy),
        'test_loss': float(loss),
        'dataset_sha256': sha256(args.dataset),
        'model_sha256': sha256(destination),
    }
    destination.with_suffix('.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    save_environment(destination.with_name('classifier_environment.json'))
    print(f'Saved {destination}. Within-dataset test accuracy: {accuracy:.4%}')


def run_residual_training(args):
    import numpy as np
    from acoustic_masking.config import read_config
    from acoustic_masking.data import load_dataset,batches,load_carrier_splits
    from acoustic_masking.models import load_frozen
    from acoustic_masking.preprocessing import LogMel
    from acoustic_masking.training import train_residual,train_original_uap
    from acoustic_masking.artifacts import save_run
    from acoustic_masking.reproducibility import set_seed,save_environment

    cfg = read_config(args.config)
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    data = load_dataset(cfg['dataset'])
    indices = data['train_idx']
    # New deterministic dataset object for every run. Notebook used shuffle(seed=42).
    dataset = batches(
        data['waveforms'][indices], data['labels'][indices], batch_size=int(cfg.get('batch_size', 32)),
        seed=int(cfg.get('data_seed', 42)), shuffle=True,
        reshuffle_each_iteration=bool(cfg.get('reshuffle_each_iteration', True)),
    )
    model = load_frozen(cfg['model'])
    log_mel = LogMel()
    kind = cfg.get('kind', 'carrier_aware')
    if kind == 'original':
        residual, history = train_original_uap(
            model, log_mel, dataset, epsilon=float(cfg.get('epsilon', 0.01)),
            step_size=float(cfg.get('step_size', 0.001)), epochs=int(cfg.get('epochs', 30)),
        )
    elif kind == 'carrier_aware':
        settings = cfg.get('carrier_config', {})
        sources = load_carrier_splits({name: cfg['carriers'][name] for name in settings})
        initial_path = cfg.get('initial_residual')
        initial = np.load(initial_path, allow_pickle=False) if initial_path else None
        residual, history = train_residual(
            model, log_mel, {name: splits['train'] for name, splits in sources.items()}, settings, dataset,
            initial_residual=initial, rms_budget=float(cfg.get('rms_budget', 0.0012)),
            peak_budget=float(cfg.get('peak_budget', 0.0024)), epochs=int(cfg.get('epochs', 320)),
            steps_per_epoch=int(cfg.get('steps_per_epoch', 12)),
            learning_rate=float(cfg.get('learning_rate', 1e-5)), seed=seed, init_mode=cfg.get('init_mode', 'zero'),
        )
    else:
        raise ValueError(f'Unknown training kind: {kind}')
    folder = save_run(
        cfg.get('output_dir', 'artifacts/runs'), cfg['run_name'], residual, history, cfg,
        model_path=cfg['model'], dataset_path=cfg['dataset'],
    )
    save_environment(folder / 'environment.json')
    print(f'Saved run: {folder}')


def load_residuals(config):
    import numpy as np
    from acoustic_masking.audio import scale_to_budget

    residuals = {}
    for name, entry in config['residuals'].items():
        if 'path' in entry:
            waveform = np.load(entry['path'], allow_pickle=False)
        else:
            with np.load(entry['bank'], allow_pickle=False) as bank:
                waveform = bank[entry['key']]
        if entry.get('match_budget', False):
            waveform = scale_to_budget(
                waveform, config.get('noise_rms', 0.0012), config.get('noise_peak', 0.0024)
            )
        residuals[name] = np.asarray(waveform, np.float32).reshape(-1)
    return residuals


def evaluate(args):
    from acoustic_masking.config import read_config
    from acoustic_masking.data import load_dataset,batches,load_carrier_splits
    from acoustic_masking.models import load_frozen
    from acoustic_masking.preprocessing import LogMel
    from acoustic_masking.evaluation import run_paired_evaluation,prediction_concentration
    from acoustic_masking.reproducibility import set_seed,sha256,save_environment

    cfg = read_config(args.config);seed=int(cfg.get('seed',20261116));set_seed(seed)
    data = load_dataset(cfg['dataset'])
    split = cfg.get('keystroke_split', 'test')
    indices = data[f'{split}_idx']
    dataset = batches(
        data['waveforms'][indices], data['labels'][indices], batch_size=int(cfg.get('batch_size', 32))
    )
    model = load_frozen(cfg['model'])
    log_mel = LogMel()
    carrier_splits = load_carrier_splits(cfg['carriers'])
    predictions, repeats, summary = run_paired_evaluation(
        model, log_mel, list(dataset), carrier_splits, cfg['carrier_config'], load_residuals(cfg),
        repeats=int(cfg.get('repeats', 30)),
        seed=seed,
        playback_scale=float(cfg.get('playback_scale', 0.90)),
        noise_rms=float(cfg.get('noise_rms', 0.0012)),
        noise_peak=float(cfg.get('noise_peak', 0.0024)),
        split=cfg.get('carrier_split', 'val'),
        clip_mixtures=bool(cfg.get('clip_mixtures', False)),
    )
    out = Path(cfg.get('output_dir', 'artifacts/evaluation'))
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / 'predictions.csv', index=False)
    repeats.to_csv(out / 'accuracy_by_repeat.csv', index=False)
    summary.to_csv(out / 'accuracy_summary.csv', index=False)
    prediction_concentration(predictions).to_csv(out / 'prediction_concentration.csv', index=False)
    manifest = {
        'config': cfg,
        'dataset_sha256': sha256(cfg['dataset']),
        'model_sha256': sha256(cfg['model']),
        'note': 'Repeated carrier/phase trials are not independent keystroke datasets',
    }
    (out / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_environment(out / 'environment.json')
    print(summary.to_string(index=False))


def plot_results(args):
    from acoustic_masking.figures import plot_accuracy,plot_top_errors

    directory = Path(args.evaluation_dir)
    plot_accuracy(directory / 'accuracy_summary.csv', directory / 'accuracy_chart.png')
    plot_top_errors(
        directory / 'predictions.csv', args.carrier, output=directory / f'errors_{args.carrier}.png'
    )
    print(f'Plots saved to {directory}')


def extract_saved_residual(args):
    import numpy as np

    with np.load(args.bank, allow_pickle=False) as bank:
        if args.key not in bank.files:
            raise KeyError(f'{args.key} not in {bank.files}')
        waveform = bank[args.key]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, waveform)
    print(f'Saved {args.key} to {path}')


def run_mask_bank(args):
    from .mask_bank import run

    run(args.stage, args.config, workers=args.workers)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m acoustic_masking", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("prepare-dataset", help="Prepare the labeled keystroke dataset")
    command.add_argument('--audio-dir', required=True)
    command.add_argument('--out', default='artifacts/models/keystroke_dataset_new.npz')
    command.add_argument('--seed', type=int, default=42)
    command.set_defaults(run=prepare_dataset)

    command = commands.add_parser("train-classifier", help="Train the keystroke classifier")
    command.add_argument('--dataset', required=True)
    command.add_argument('--out', default='artifacts/models/keystroke_cnn_new.keras')
    command.add_argument('--seed', type=int, default=42)
    command.add_argument('--epochs', type=int, default=30)
    command.add_argument('--batch-size', type=int, default=32)
    command.set_defaults(run=train_classifier)

    command = commands.add_parser("train-residual", help="Train a masking residual")
    command.add_argument('--config', required=True)
    command.set_defaults(run=run_residual_training)

    command = commands.add_parser("evaluate", help="Evaluate masks on paired carrier windows")
    command.add_argument('--config', required=True)
    command.set_defaults(run=evaluate)

    command = commands.add_parser("plot-results", help="Plot paired evaluation results")
    command.add_argument('--evaluation-dir', default='artifacts/evaluation')
    command.add_argument('--carrier', default='fan')
    command.set_defaults(run=plot_results)

    command = commands.add_parser("extract-saved-residual", help="Extract a waveform from the residual bank")
    command.add_argument('--bank', required=True)
    command.add_argument('--key', required=True)
    command.add_argument('--out', required=True)
    command.set_defaults(run=extract_saved_residual)

    command = commands.add_parser("mask-bank", help="Train and evaluate a residual bank")
    command.add_argument("stage", choices=["train", "evaluate-zero", "evaluate"])
    command.add_argument("--config", default="configs/mask_bank_pilot.yaml")
    command.add_argument("--workers", type=int, choices=[1, 2], default=1,
                         help="Train full-bank initialization groups in separate processes")
    command.set_defaults(run=run_mask_bank)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
