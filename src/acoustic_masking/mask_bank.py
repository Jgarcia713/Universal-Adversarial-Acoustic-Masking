"""Train and evaluate independently learned masking residuals for a pilot bank.

The same sampling seeds are used across initialization modes. The pilot
tracks waveform, spectrum, and paired prediction differences and reports
the accuracy cost of reducing residual gain underneath each carrier.
"""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from .audio import project_residual
from .config import N_SAMPLES, SAMPLE_RATE, read_config
from .reproducibility import save_environment, set_seed, sha256


def candidates(config):
    """Yield candidate identifiers in training order, with zero starts first."""
    for mode in ("zero", "random"):
        seeds = config.get("seeds_by_initialization", {}).get(mode, config.get("seeds", []))
        for seed in seeds:
            yield mode, seed, f"{mode}_{seed}"


def train(config, initialization=None):
    """Create or reuse one complete training run per initialization and seed."""
    from .artifacts import save_run
    from .data import batches, load_carrier_splits, load_dataset
    from .models import load_frozen
    from .preprocessing import LogMel
    from .training import train_residual

    settings = read_config(config["training_config"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    dataset = load_dataset(settings["dataset"])
    model = load_frozen(settings["model"])
    features = LogMel()
    sources = {
        name: splits["train"]
        for name, splits in load_carrier_splits(settings["carriers"]).items()
    }
    indices = dataset["train_idx"]
    checkpoints = config.get("checkpoints", [])
    if checkpoints and sorted(set(checkpoints)) != checkpoints:
        raise ValueError("checkpoints must be distinct and in increasing order")
    if any(not 1 <= epoch <= config["epochs"] for epoch in checkpoints):
        raise ValueError("checkpoints must be within the training horizon")
    input_hashes = {
        path: sha256(path)
        for path in (settings["model"], settings["dataset"], *settings["carriers"].values())
    }
    if config.get("export_candidates"):
        freeze = {
            "bank_config": config,
            "training_settings": settings,
            "input_sha256": input_hashes,
            "trainer_sha256": sha256(Path(__file__).with_name("training.py")),
        }
        freeze_path = output / "frozen_training.json"
        if freeze_path.exists():
            if json.loads(freeze_path.read_text(encoding="utf-8")) != freeze:
                raise ValueError("Training settings, inputs, or trainer changed since the bank was frozen")
        else:
            with TemporaryDirectory(dir=output) as temporary:
                staged = Path(temporary) / freeze_path.name
                staged.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
                staged.replace(freeze_path)

    for mode, seed, name in candidates(config):
        if initialization is not None and mode != initialization:
            continue
        folder = output / name
        run = {
            **settings,
            "run_name": name,
            "output_dir": str(output),
            "epochs": config["epochs"],
            "seed": seed,
            "data_seed": seed,
            "init_mode": mode,
            "initialization_rms": config["initialization_rms"] if mode == "random" else 0.0,
            "compiled_step": True,
            "input_sha256": input_hashes,
        }
        if checkpoints:
            run["checkpoints"] = checkpoints
        if folder.exists():
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["config"] != run
                    or manifest["residual_sha256"] != sha256(folder / "residual.npy")
                    or any(manifest.get("checkpoint_sha256", {}).get(str(epoch))
                           != sha256(folder / f"epoch_{epoch}.npy") for epoch in checkpoints)
                    or not (folder / "history.csv").is_file()):
                raise ValueError(f"Saved run does not match this pilot: {folder}")
            print(f"Reusing {folder}", flush=True)
            continue
        if config.get("reuse_from"):
            source = Path(config["reuse_from"]) / name
            if source.is_dir():
                reuse_run(source, folder, run)
                print(f"Reused {source}", flush=True)
                continue
        set_seed(seed)
        data = batches(
            dataset["waveforms"][indices], dataset["labels"][indices],
            batch_size=settings["batch_size"], seed=seed,
            shuffle=True, reshuffle_each_iteration=settings["reshuffle_each_iteration"],
        )
        initial = None
        if mode == "random":
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            initial = project_residual(
                rng.normal(0, config["initialization_rms"], N_SAMPLES).astype(np.float32),
                config["initialization_rms"], settings["peak_budget"],
            )
        print(f"Training {name}", flush=True)
        snapshots = {}

        def checkpoint(epoch, wave):
            if epoch in checkpoints:
                snapshots[epoch] = wave.copy()

        waveform, history = train_residual(
            model, features, sources, settings["carrier_config"], data,
            initial_residual=initial, rms_budget=settings["rms_budget"],
            peak_budget=settings["peak_budget"], epochs=config["epochs"],
            steps_per_epoch=settings["steps_per_epoch"], learning_rate=settings["learning_rate"],
            seed=seed, compile_step=True, on_epoch_end=checkpoint if checkpoints else None,
        )
        with TemporaryDirectory(dir=output) as temporary:
            saved = save_run(temporary, name, waveform, history, run, settings["model"], settings["dataset"])
            if snapshots:
                manifest_path = saved / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["checkpoint_sha256"] = {}
                for epoch, snapshot in snapshots.items():
                    path = saved / f"epoch_{epoch}.npy"
                    np.save(path, snapshot)
                    manifest["checkpoint_sha256"][str(epoch)] = sha256(path)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            save_environment(saved / "environment.json")
            saved.rename(folder)
        print(f"Saved {folder}", flush=True)
    if config.get("export_candidates") and initialization is None:
        export_candidates(config, output)


def reuse_run(source, destination, expected):
    """Copy a completed waveform whose training configuration matches exactly."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    original = dict(manifest["config"])
    original.pop("checkpoints", None)
    original["output_dir"] = expected["output_dir"]
    if original != expected or sha256(source / "residual.npy") != manifest["residual_sha256"]:
        raise ValueError(f"Reusable run does not match the frozen configuration: {source}")
    manifest["config"] = expected
    manifest["reused_from"] = str(source)
    manifest.pop("checkpoint_sha256", None)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        for name in ("residual.npy", "history.csv", "environment.json"):
            copyfile(source / name, staging / name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(destination)


def export_candidates(config, output):
    """Package all completed bank candidates and record their run provenance."""
    waves = {}
    entries = {}
    for mode, seed, name in candidates(config):
        folder = output / name
        path = folder / "residual.npy"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        digest = sha256(path)
        if digest != manifest["residual_sha256"]:
            raise ValueError(f"Residual hash mismatch: {path}")
        wave = np.load(path, allow_pickle=False)
        if wave.shape != (N_SAMPLES,) or not np.isfinite(wave).all():
            raise ValueError(f"Invalid residual waveform: {path}")
        if (np.max(np.abs(wave)) > manifest["config"]["peak_budget"] + 1e-8
                or np.sqrt(np.mean(wave ** 2)) > manifest["config"]["rms_budget"] + 1e-8):
            raise ValueError(f"Residual exceeds its training budget: {path}")
        waves[name] = wave
        entries[name] = {
            "initialization": mode, "seed": seed, "epochs": config["epochs"],
            "residual_sha256": digest, "run": str(folder),
            "reused_from": manifest.get("reused_from"),
        }
    if len({entry["residual_sha256"] for entry in entries.values()}) != len(entries):
        raise ValueError("The candidate bank contains identical waveforms")
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        bank = staging / "candidate_bank.npz"
        np.savez_compressed(bank, **waves)
        metadata = {
            "sample_rate_hz": SAMPLE_RATE, "samples_per_waveform": N_SAMPLES,
            "candidate_bank_sha256": sha256(bank), "candidates": entries,
        }
        manifest_path = staging / "candidate_bank_manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        bank.replace(output / bank.name)
        manifest_path.replace(output / manifest_path.name)
    print(f"Exported {len(entries)} candidates to {output / bank.name}", flush=True)


def evaluate(config, zero_only=False, checkpoint=None):
    """Score the scratch references and pilot candidates on common trials."""
    import tensorflow as tf
    from .data import batches, load_carrier_splits, load_dataset
    from .evaluation import run_paired_evaluation
    from .models import load_frozen
    from .preprocessing import LogMel

    settings = read_config(config["evaluation_config"])
    output = Path(config["output_dir"])
    reference = settings["residuals"][config["reference_key"]]
    with np.load(reference["bank"], allow_pickle=False) as bank:
        waveforms = {"saved_scratch": bank[reference["key"]]}
    if config.get("reference_run"):
        waveforms["previous_retrain"] = np.load(config["reference_run"], allow_pickle=False)
    names = [name for mode, _, name in candidates(config) if not zero_only or mode == "zero"]
    filename = "residual.npy" if checkpoint is None else f"epoch_{checkpoint}.npy"
    for name in names:
        folder = output / name
        path = folder / filename
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        expected_hash = (manifest["residual_sha256"] if checkpoint is None
                         else manifest["checkpoint_sha256"][str(checkpoint)])
        if expected_hash != sha256(path):
            raise ValueError(f"Residual hash mismatch: {path}")
        if manifest["config"]["epochs"] != config["epochs"]:
            raise ValueError(f"Unexpected training horizon for {name}")
        waveforms[name] = np.load(path, allow_pickle=False)

    for name, wave in waveforms.items():
        if wave.shape != (N_SAMPLES,) or not np.isfinite(wave).all() or np.std(wave) == 0:
            raise ValueError(f"Invalid residual waveform: {name}")
    conditions = {
        f"{name}@{scale:g}": wave * scale
        for name, wave in waveforms.items()
        for scale in ([1.0] if zero_only else config["loudness_scales"])
    }
    set_seed(settings["seed"])
    dataset = load_dataset(settings["dataset"])
    indices = dataset[f'{settings["keystroke_split"]}_idx']
    data = list(batches(dataset["waveforms"][indices], dataset["labels"][indices],
                        batch_size=settings["batch_size"]))
    model = load_frozen(settings["model"])
    features = LogMel()

    @tf.function(reduce_retracing=True)
    def predict(audio, training=False):
        return model(features(audio), training=training)

    carriers = load_carrier_splits(settings["carriers"])
    predictions, repeats, summary = run_paired_evaluation(
        predict, tf.identity, data, carriers, settings["carrier_config"], conditions,
        repeats=config["evaluation_repeats"], seed=settings["seed"],
        playback_scale=settings["playback_scale"], noise_rms=settings["noise_rms"],
        noise_peak=settings["noise_peak"], split=settings["carrier_split"],
        clip_mixtures=settings["clip_mixtures"],
    )
    destination = output / (
        f"evaluation_epoch_{checkpoint}" if checkpoint is not None
        else "zero_evaluation" if zero_only else "evaluation"
    )
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "predictions.csv.gz", index=False)
    repeats.to_csv(destination / "accuracy_by_repeat.csv", index=False)
    summary.to_csv(destination / "accuracy_summary.csv", index=False)
    save_environment(destination / "environment.json")
    (destination / "config.json").write_text(
        json.dumps({
            "pilot": config,
            "evaluation": settings,
            "checkpoint": checkpoint,
            "input_sha256": {
                path: sha256(path) for path in (
                    settings["model"], settings["dataset"], reference["bank"],
                    *settings["carriers"].values(),
                    *([config["reference_run"]] if config.get("reference_run") else []),
                    *(str(output / name / filename) for name in names),
                )
            },
        }, indent=2), encoding="utf-8"
    )
    quiet_window_rms = {}
    for name, carrier in settings["carrier_config"].items():
        source = carriers[name][settings["carrier_split"]].astype(np.float64)
        cumulative_energy = np.concatenate(([0.0], np.cumsum(source ** 2)))
        starts = predictions.loc[
            predictions["carrier"].eq(name) & predictions["condition"].eq("carrier_only"),
            "carrier_start",
        ].to_numpy()
        window_rms = np.sqrt(np.maximum(
            (cumulative_energy[starts + N_SAMPLES] - cumulative_energy[starts]) / N_SAMPLES, 0
        ))
        quiet_window_rms[name] = (
            np.percentile(window_rms, 5) * carrier["carrier_rms"] / np.sqrt(np.mean(source ** 2))
        )
    analyze(config, settings, waveforms, names, predictions, summary, destination, quiet_window_rms, zero_only)
    return destination


def analyze(config, settings, waveforms, names, predictions, summary, destination, quiet_window_rms,
            zero_only=False):
    """Write diversity and gain measurements and export passing waveforms."""
    import pandas as pd
    from scipy.spatial.distance import jensenshannon

    from .audio import rms

    trial = ["carrier", "repeat", "sample_id"]
    groups = predictions.groupby(trial, sort=False)
    audit = groups[["carrier_start", "residual_phase", "true_label"]].nunique()
    if not audit.eq(1).all().all():
        raise ValueError("Evaluation conditions used different trials")
    paired = predictions.pivot(index=trial, columns="condition", values="predicted_label")
    if paired.isna().any().any():
        raise ValueError("Incomplete paired predictions")
    labels = list(waveforms)
    full_predictions = paired[[f"{name}@1" for name in labels]].to_numpy()
    matrix = {metric: pd.DataFrame(0.0, index=labels, columns=labels) for metric in (
        "phase_max_abs_correlation", "spectral_js_distance", "prediction_disagreement",
    )}
    centered = {name: wave - wave.mean() for name, wave in waveforms.items()}
    transforms = {name: np.fft.rfft(wave) for name, wave in centered.items()}
    for i, left in enumerate(labels):
        for j in range(i, len(labels)):
            right = labels[j]
            cross = np.fft.irfft(transforms[left].conj() * transforms[right], n=N_SAMPLES)
            norm = np.linalg.norm(centered[left]) * np.linalg.norm(centered[right])
            values = (
                min(1.0, float(np.max(np.abs(cross)) / norm)),
                float(jensenshannon(abs(transforms[left]) ** 2, abs(transforms[right]) ** 2, base=2)),
                float(np.mean(full_predictions[:, i] != full_predictions[:, j])),
            )
            for frame, value in zip(matrix.values(), values):
                frame.loc[left, right] = frame.loc[right, left] = value
    for metric, frame in matrix.items():
        frame.to_csv(destination / f"{metric}.csv", index_label="mask")
    plot_diversity(matrix, destination / "diversity_matrix.png")

    group_rows = []
    for mode in ("zero", "random"):
        members = [name for scheme, _, name in candidates(config) if scheme == mode and name in names]
        if len(members) < 2:
            continue
        off_diagonal = np.triu_indices(len(members), k=1)
        for metric, frame in matrix.items():
            values = frame.loc[members, members].to_numpy()[off_diagonal]
            group_rows.append({
                "initialization": mode, "metric": metric,
                "min": values.min(), "median": np.median(values), "max": values.max(),
            })
    pd.DataFrame(group_rows).to_csv(destination / "diversity_by_initialization.csv", index=False)

    accuracy = summary.pivot(index="carrier", columns="condition", values="mean")
    reference_accuracy = accuracy["saved_scratch@1"]
    rows = []
    selected = {}
    for name, wave in waveforms.items():
        for scale in ([1.0] if zero_only else config["loudness_scales"]):
            scores = accuracy[f"{name}@{scale:g}"]
            cost = (scores - accuracy[f"{name}@1"]) * 100
            reference_delta = (scores - reference_accuracy) * 100
            passed = bool(reference_delta.le(config["accuracy_tolerance_pp"]).all())
            if name in names and passed:
                selected[name] = min(scale, selected.get(name, scale))
            for carrier, carrier_settings in settings["carrier_config"].items():
                residual_rms = rms(wave) * scale * settings["playback_scale"] * carrier_settings["residual_gain"]
                rows.append({
                    "mask": name, "scale": scale, "carrier": carrier,
                    "residual_to_carrier_rms_db": 20 * np.log10(residual_rms / carrier_settings["carrier_rms"]),
                    "residual_to_quiet_window_rms_db": (
                        20 * np.log10(residual_rms / quiet_window_rms[carrier])
                        if quiet_window_rms[carrier] > 0 else np.inf
                    ),
                    "accuracy_pct": 100 * scores[carrier],
                    "cost_vs_full_pp": cost[carrier],
                    "delta_vs_saved_scratch_pp": reference_delta[carrier],
                    "passes_all_carriers": passed,
                })
    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(destination / "loudness_tradeoff.csv", index=False)
    if zero_only:
        print(accuracy.to_string())
        return
    initialization = tradeoff.loc[tradeoff["mask"].isin(names)].copy()
    initialization[["initialization", "seed"]] = initialization["mask"].str.split("_", expand=True)
    paired_accuracy = initialization.pivot(
        index=["seed", "scale", "carrier"], columns="initialization", values="accuracy_pct"
    ).dropna(subset=["random", "zero"])
    paired_accuracy["random_minus_zero_pp"] = paired_accuracy["random"] - paired_accuracy["zero"]
    paired_accuracy.to_csv(destination / "paired_initialization_accuracy.csv")

    admitted, duplicates = screen_duplicates(config, selected, matrix)

    candidate_path = destination / "candidate_bank.npz"
    np.savez_compressed(candidate_path, **{name: waveforms[name] for name in names})
    bank_path = destination / "pilot_bank.npz"
    np.savez_compressed(
        bank_path,
        **{name: waveforms[name] * scale for name, scale in admitted.items()},
    )
    outcome = {
        "matched_trials": len(paired),
        "sample_rate_hz": SAMPLE_RATE,
        "samples_per_waveform": N_SAMPLES,
        "candidate_bank_sha256": sha256(candidate_path),
        "bank_sha256": sha256(bank_path),
        "accuracy_tolerance_pp_per_carrier": config["accuracy_tolerance_pp"],
        "passing_scales": selected,
        "rejected_for_similarity": duplicates,
        "bank_waveforms_with_scale_already_applied": admitted,
        "paired_setup": {
            "playback_scale": settings["playback_scale"],
            "carrier_config": settings["carrier_config"],
        },
        "min_admitted_for_expansion": config["min_admitted_for_expansion"],
        "expansion_supported": len(admitted) >= config["min_admitted_for_expansion"],
    }
    (destination / "selection.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))


def screen_duplicates(config, selected, matrix):
    """Prefer quieter passing masks, then remove candidates similar to ones kept."""
    admitted = {}
    duplicates = {}
    for name in sorted(selected, key=lambda name: (selected[name], name)):
        for existing in admitted:
            correlation = matrix["phase_max_abs_correlation"].loc[name, existing]
            distance = matrix["spectral_js_distance"].loc[name, existing]
            disagreement = matrix["prediction_disagreement"].loc[name, existing]
            if (correlation >= config["duplicate_correlation"]
                    or (distance <= config["duplicate_spectral_distance"]
                        and disagreement <= config["duplicate_prediction_disagreement"])):
                duplicates[name] = existing
                break
        else:
            admitted[name] = selected[name]
    return admitted, duplicates


def plot_diversity(matrix, path):
    import matplotlib.pyplot as plt

    titles = ("Peak circular correlation", "Spectral Jensen–Shannon distance", "Prediction disagreement")
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    for ax, title, frame in zip(axes, titles, matrix.values()):
        values = frame.to_numpy(copy=True)
        np.fill_diagonal(values, np.nan)
        heatmap = ax.imshow(values, cmap="viridis")
        positions = np.arange(len(frame))
        ax.set_xticks(positions, frame.columns, rotation=65, ha="right")
        ax.set_yticks(positions, frame.index)
        ax.set_title(title)
        figure.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate_checkpoints(config):
    """Compare checkpoints on the pilot's held-fixed carrier and phase draws."""
    import pandas as pd

    output = Path(config["output_dir"])
    pilot = Path(config["pilot_output_dir"])
    epochs = config["checkpoints"]
    names = [name for _, _, name in candidates(config)]
    pilot_eval = pilot / "evaluation"
    pilot_record = json.loads((pilot_eval / "config.json").read_text(encoding="utf-8"))
    settings = read_config(config["evaluation_config"])
    pilot_settings = pilot_record["pilot"]
    if (pilot_record["evaluation"] != settings
            or any(pilot_settings.get(key) != config.get(key) for key in (
                "seeds", "initialization_rms", "evaluation_repeats", "reference_key",
                "reference_run", "accuracy_tolerance_pp", "duplicate_correlation",
                "duplicate_spectral_distance", "duplicate_prediction_disagreement",
                "min_admitted_for_expansion",
            ))
            or epochs[0] != pilot_settings["epochs"]):
        raise ValueError("Checkpoint and pilot evaluation settings differ")
    for path, digest in pilot_record["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Pilot input has changed: {path}")

    replay = []
    for name in names:
        earlier = json.loads((pilot / name / "manifest.json").read_text(encoding="utf-8"))
        later = json.loads((output / name / "manifest.json").read_text(encoding="utf-8"))
        exclude = {"output_dir", "epochs", "checkpoints"}
        if ({k: v for k, v in earlier["config"].items() if k not in exclude}
                != {k: v for k, v in later["config"].items() if k not in exclude}):
            raise ValueError(f"Training settings changed for {name}")
        previous = np.load(pilot / name / "residual.npy", allow_pickle=False)
        path = output / name / f"epoch_{epochs[0]}.npy"
        if sha256(path) != later["checkpoint_sha256"][str(epochs[0])]:
            raise ValueError(f"Residual hash mismatch: {path}")
        checkpoint = np.load(path, allow_pickle=False)
        replay.append({
            "mask": name, "exact_waveform_match": bool(np.array_equal(previous, checkpoint)),
            "max_abs_difference": float(np.max(np.abs(previous - checkpoint))),
        })
    replay = pd.DataFrame(replay)
    replay.to_csv(output / "pilot_replay.csv", index=False)
    exact = bool(replay["exact_waveform_match"].all())
    print(f"Exact pilot replays: {replay['exact_waveform_match'].sum()}/{len(names)}", flush=True)

    scores = []
    diversity = []
    selection = {}
    trial_draws = None
    for epoch in epochs:
        if epoch == epochs[0] and exact:
            destination = pilot_eval
        else:
            print(f"Evaluating epoch {epoch}", flush=True)
            destination = evaluate(config, checkpoint=epoch)
        predictions = pd.read_csv(destination / "predictions.csv.gz")
        index = ["carrier", "repeat", "sample_id"]
        draws = predictions.loc[predictions["condition"].eq("saved_scratch@1")].set_index(index)
        draws = draws[["carrier_start", "residual_phase", "true_label", "predicted_label"]].sort_index()
        if trial_draws is None:
            trial_draws = draws
        else:
            pd.testing.assert_frame_equal(trial_draws, draws)
        full_gain = pd.read_csv(destination / "loudness_tradeoff.csv").query("scale == 1").assign(epoch=epoch)
        scores.append(full_gain)
        diversity.append(pd.read_csv(destination / "diversity_by_initialization.csv").assign(epoch=epoch))
        matrices = {
            metric: pd.read_csv(destination / f"{metric}.csv", index_col="mask")
            for metric in ("phase_max_abs_correlation", "spectral_js_distance", "prediction_disagreement")
        }
        passing = {name: 1.0 for name in full_gain.loc[full_gain["passes_all_carriers"], "mask"] if name in names}
        admitted, _ = screen_duplicates(config, passing, matrices)
        selection[epoch] = sorted(admitted)

    accuracy = pd.concat(scores, ignore_index=True)
    accuracy.to_csv(output / "checkpoint_accuracy.csv", index=False)
    pd.concat(diversity, ignore_index=True).to_csv(output / "checkpoint_diversity.csv", index=False)
    trajectory = accuracy.loc[accuracy["mask"].isin(names)].pivot(
        index=["mask", "carrier"], columns="epoch", values="accuracy_pct"
    )
    for earlier, later in zip(epochs, epochs[1:]):
        trajectory[f"change_{earlier}_to_{later}_pp"] = trajectory[later] - trajectory[earlier]
    trajectory.to_csv(output / "paired_checkpoint_changes.csv")
    plot_checkpoint_accuracy(accuracy, names, epochs, tolerance_pp=config["accuracy_tolerance_pp"],
                             path=output / "checkpoint_accuracy.png")
    _, intermediate, final = epochs
    tolerance = config["near_equivalence_pp"]
    penalty = trajectory[intermediate] - trajectory[final]
    outcome = {
        "matched_trials": len(trial_draws),
        "reused_original_pilot_evaluation": exact,
        "admitted_masks_by_epoch": selection,
        "intermediate_worst_accuracy_penalty_pp": float(penalty.max()),
        "intermediate_tolerance_pp": tolerance,
        "intermediate_retains_final_admissions": set(selection[final]).issubset(selection[intermediate]),
        "intermediate_within_tolerance_everywhere": bool(penalty.le(tolerance).all()),
    }
    (output / "checkpoint_summary.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))


def plot_checkpoint_accuracy(accuracy, names, epochs, tolerance_pp, path):
    """Plot each trajectory with the saved scratch accuracy and admission cutoff."""
    import matplotlib.pyplot as plt

    carriers = sorted(accuracy["carrier"].unique())
    fig, axes = plt.subplots(1, len(carriers), figsize=(12, 4.2), layout="constrained", squeeze=False)
    colors = {"zero": "#255c8d", "random": "#a65127"}
    for ax, carrier in zip(axes[0], carriers):
        scores = accuracy.loc[accuracy["carrier"].eq(carrier)].pivot(
            index="mask", columns="epoch", values="accuracy_pct"
        )
        reference = scores.loc["saved_scratch", epochs[0]]
        for mode, color in colors.items():
            members = [name for name in names if name.startswith(f"{mode}_")]
            for _, row in scores.loc[members, epochs].iterrows():
                ax.plot(epochs, row, color=color, alpha=0.32, linewidth=1)
            ax.plot(epochs, scores.loc[members, epochs].mean(), "o-", color=color,
                    label=f"{mode.capitalize()} mean")
        ax.axhline(reference, linestyle=":", color="#333333", label="Saved scratch")
        ax.axhline(reference + tolerance_pp, linestyle="--", color="#666666", label="Admission cutoff")
        ax.set(title=carrier.capitalize(), xlabel="Training epoch", xticks=epochs)
    axes[0, 0].set_ylabel("Classifier accuracy (%)")
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(stage, config_path, workers=1):
    config = read_config(config_path)
    for mode in ("zero", "random"):
        seeds = [seed for scheme, seed, _ in candidates(config) if scheme == mode]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError(f"{mode} must have at least two distinct seeds")
    scales = config["loudness_scales"]
    if 1.0 not in scales or any(not 0 < scale <= 1.0 for scale in scales) or len(set(scales)) != len(scales):
        raise ValueError("loudness_scales must contain distinct positive gains up to and including 1.0")
    if workers == 2:
        if stage != "train" or not config.get("export_candidates"):
            raise ValueError("Two workers require a full candidate bank training run")
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
            work = [pool.submit(train, config, mode) for mode in ("zero", "random")]
            for worker in work:
                worker.result()
        export_candidates(config, Path(config["output_dir"]))
    elif stage == "train":
        train(config)
    elif config.get("checkpoints"):
        if stage != "evaluate" or len(config["checkpoints"]) != 3 or scales != [1.0]:
            raise ValueError("Checkpoint evaluation needs three epochs, full gain, and the evaluate stage")
        evaluate_checkpoints(config)
    else:
        evaluate(config, zero_only=stage == "evaluate-zero")
