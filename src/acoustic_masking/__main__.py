"""Command-line entry point for the acoustic masking experiments.

Dispatches commands to prepare data, train the classifier or a masking
residual, evaluate saved waveforms, compare classifiers on clean keystrokes
and under masking, plot results, and extract a bank entry.
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
    from acoustic_masking.models import ARCHITECTURES
    from acoustic_masking.reproducibility import set_seed,save_environment,sha256

    set_seed(args.seed)
    data = load_dataset(args.dataset)
    features = np.stack([librosa_log_mel(wave) for wave in data['waveforms']])[..., None]
    labels = data['labels']
    train, val, test = (data[key] for key in ('train_idx', 'val_idx', 'test_idx'))
    model = ARCHITECTURES[args.arch]()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(destination.with_name(f'{destination.stem}_best.keras')),
            monitor='val_loss', save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(destination.with_name(f'{destination.stem}_history.csv'))),
    ]
    history = model.fit(
        features[train], labels[train], validation_data=(features[val], labels[val]),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1,
    )
    model.save(destination)
    loss, accuracy = model.evaluate(features[test], labels[test], verbose=0)
    metrics = {
        'arch': args.arch,
        'seed': args.seed,
        'max_epochs': args.epochs,
        'epochs_run': len(history.history['loss']),
        'batch_size': args.batch_size,
        'parameters': int(model.count_params()),
        'test_accuracy': float(accuracy),
        'test_loss': float(loss),
        'dataset_sha256': sha256(args.dataset),
        'model_sha256': sha256(destination),
    }
    destination.with_suffix('.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    # Named per model so training a second classifier doesn't overwrite another's record.
    save_environment(destination.with_name(f'{destination.stem}_environment.json'))
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


def evaluate_clean(args):
    import itertools
    import pandas as pd
    from sklearn.metrics import f1_score
    from acoustic_masking.config import CLASSES
    from acoustic_masking.data import load_dataset
    from acoustic_masking.models import load_frozen
    from acoustic_masking.preprocessing import LogMel
    from acoustic_masking.evaluation import clean_predictions,confusion_matrix_frame,paired_clean_comparison
    from acoustic_masking.reproducibility import sha256,save_environment

    models = {}
    for spec in args.model:
        name, sep, path = spec.partition('=')
        if not sep or not name or not path: raise ValueError(f'--model must be NAME=PATH, got {spec!r}')
        if name in models: raise ValueError(f'Duplicate model name {name!r}')
        models[name] = path
    data = load_dataset(args.dataset)
    indices = data[f'{args.split}_idx']
    filenames = data.get('filenames')
    if filenames is not None:
        stems = [Path(str(f).replace('\\', '/')).name.split('_')[0] for f in filenames[indices]]
        mismatched = sum(stem != CLASSES[label] for stem, label in zip(stems, data['labels'][indices]))
        if mismatched: raise ValueError(f'{mismatched} labels disagree with their filenames')
    log_mel = LogMel()
    frames, summary = [], []
    for name, path in models.items():
        frame = clean_predictions(load_frozen(path), log_mel, data['waveforms'], data['labels'],
                                  indices, name, filenames)
        frames.append(frame)
        summary.append({'model_id': name, 'model_path': path, 'model_sha256': sha256(path),
                        'split': args.split, 'n_examples': len(frame), 'n_correct': int(frame.correct.sum()),
                        'accuracy': float(frame.correct.mean()),
                        'macro_f1': float(f1_score(frame.true_label, frame.predicted_label,
                                                   labels=range(len(CLASSES)), average='macro'))})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(out / 'clean_predictions.csv', index=False)
    for frame in frames:
        confusion_matrix_frame(frame.true_label, frame.predicted_label).to_csv(
            out / f'confusion_{frame.model_id.iloc[0]}.csv')
    pd.DataFrame(summary).to_csv(out / 'clean_summary.csv', index=False)
    pairs = []
    for a, b in itertools.combinations(frames, 2):
        if not (a.dataset_index.to_numpy() == b.dataset_index.to_numpy()).all():
            raise ValueError('Models were not evaluated on the same examples')
        pairs.append({'model_a': a.model_id.iloc[0], 'model_b': b.model_id.iloc[0],
                      **paired_clean_comparison(a.correct, b.correct)})
    if pairs: pd.DataFrame(pairs).to_csv(out / 'paired_comparison.csv', index=False)
    manifest = {'dataset': args.dataset, 'dataset_sha256': sha256(args.dataset), 'split': args.split,
                'models': models, 'features': 'acoustic_masking.preprocessing.LogMel',
                'note': 'Nominal test split was used during mask development; report as development evaluation'}
    (out / 'clean_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_environment(out / 'environment.json')
    print(pd.DataFrame(summary)[['model_id', 'n_correct', 'n_examples', 'accuracy', 'macro_f1']].to_string(index=False))
    if pairs: print(pd.DataFrame(pairs).to_string(index=False))


def summarize_transfer(args):
    import pandas as pd
    from acoustic_masking.evaluation import transfer_tables
    from acoustic_masking.figures import plot_transfer

    runs, sources = {}, {}
    for spec in args.evaluations:
        name, sep, folder = spec.partition('=')
        if not sep or not name or not folder: raise ValueError(f'--evaluation must be NAME=DIR, got {spec!r}')
        if name in runs: raise ValueError(f'Duplicate run name {name!r}')
        manifest = json.loads((Path(folder) / 'evaluation_manifest.json').read_text(encoding='utf-8'))
        predictions = pd.read_csv(Path(folder) / 'predictions.csv')
        runs[name] = (predictions, manifest['config'], manifest['model_sha256'])
        sources[name] = {'evaluation_dir': folder, 'model': manifest['config']['model'],
                         'model_sha256': manifest['model_sha256']}
    if args.reference and args.reference not in runs: raise ValueError(f'Unknown reference {args.reference!r}')
    by_repeat, summary, comparison = transfer_tables(runs, reference=args.reference)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_repeat.to_csv(out / 'mask_transfer_by_carrier.csv', index=False)
    summary.to_csv(out / 'transfer_summary.csv', index=False)
    comparison.to_csv(out / 'transfer_comparison.csv', index=False)
    labels = {'cnn': 'Original CNN (masks optimised against it)', 'factored': 'Factored CNN (new, unseen by masks)'}
    plot_transfer(out / 'transfer_comparison.csv', out / 'transfer_by_carrier.png', model_labels=labels)
    reference = args.reference or next(iter(runs))
    (out / 'transfer_manifest.json').write_text(json.dumps({
        'runs': sources, 'reference_model': reference,
        'paired': 'Verified identical keystrokes, carrier windows and residual phases across runs',
        'drop_definition': 'carrier_only accuracy minus condition accuracy, within the same repeat',
        'transfer_ratio': f'model drop / {reference} drop (blank when the reference drop <= 0)',
        'note': 'Repeats vary carrier windows and phases over the same keystrokes; they are repeated measures',
    }, indent=2), encoding='utf-8')
    show = comparison[comparison.condition.isin(['carrier_only', 'random_noise', 'original_uap', 'multi_scratch', 'bank_mean'])]
    print(show.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'Wrote {out}')


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
    command.add_argument('--arch', choices=['cnn', 'factored'], default='cnn',
                         help="'cnn' = original small CNN; 'factored' = factored-convolution CNN from the notebook")
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

    command = commands.add_parser("evaluate-clean", help="Compare classifiers on the same unmasked keystrokes")
    command.add_argument('--model', action='append', required=True, metavar='NAME=PATH',
                         help='Repeat once per model, e.g. --model cnn=artifacts/models/keystroke_cnn.keras')
    command.add_argument('--dataset', default='artifacts/models/keystroke_dataset.npz')
    command.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    command.add_argument('--out', required=True)
    command.set_defaults(run=evaluate_clean)

    command = commands.add_parser("summarize-transfer", help="Compare paired masking evaluations of several classifiers")
    command.add_argument('--evaluation', dest='evaluations', action='append', required=True, metavar='NAME=DIR',
                         help='Evaluation output folder; repeat per model. The first is the reference unless --reference is given')
    command.add_argument('--reference', help='Model the masks were optimised against (default: first --run)')
    command.add_argument('--out', required=True)
    command.set_defaults(run=summarize_transfer)

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
