"""Waveform operations for carrier recordings and masking residuals.

Filters and splits environmental audio, normalizes carrier sources, and
extracts windows for playback mixtures. Also provides circular shifts,
RMS and peak measurements, amplitude constraints, and random controls.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .config import N_SAMPLES, SAMPLE_RATE

def rms(audio):
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0: raise ValueError("Empty waveform")
    return float(np.sqrt(np.mean(x*x)))

def peak(audio):
    x = np.asarray(audio)
    if x.size == 0: raise ValueError("Empty waveform")
    return float(np.max(np.abs(x)))

def highpass_audio(audio, cutoff=100, sr=SAMPLE_RATE):
    sos = butter(N=4, Wn=cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(audio, dtype=np.float32)).astype(np.float32)

def split_carrier_audio(audio, train_frac=0.70, val_frac=0.15, n_samples=N_SAMPLES):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(audio)
    a, b = int(n*train_frac), int(n*(train_frac+val_frac))
    result = {"train": audio[:a], "val": audio[a:b], "test": audio[b:]}
    for name, part in result.items():
        if len(part) < n_samples: raise ValueError(f"Carrier {name} split shorter than {n_samples} samples")
    return result

def normalize_source(source, target_rms):
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    if target_rms < 0: raise ValueError("Negative carrier RMS")
    if len(source) < N_SAMPLES: raise ValueError("Carrier shorter than 0.25-second window")
    value = rms(source)
    if value <= 0: raise ValueError("Carrier must not be silent")
    return (source * (target_rms / value)).astype(np.float32)

def carrier_windows(source, starts, scale=1.0, n_samples=N_SAMPLES):
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if np.any(starts < 0) or np.any(starts+n_samples > len(source)):
        raise ValueError("Carrier window outside source")
    return (source[starts[:,None] + np.arange(n_samples)[None,:]] * scale).astype(np.float32)

def phase_shift_batch(waveform, phases, n_samples=N_SAMPLES):
    """Evaluation convention from original notebook: residual[(i-phase) % N]."""
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(wave) != n_samples: raise ValueError("Expected exactly one residual window")
    phases = np.asarray(phases, dtype=np.int64).reshape(-1)
    return wave[(np.arange(n_samples)[None,:]-phases[:,None]) % n_samples]

def scale_to_budget(waveform, target_rms=0.0012, max_peak=0.0024):
    """Uniform scaling (shape-preserving), reproducing original-UAP baseline."""
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if rms(wave) <= 0 or peak(wave) <= 0: raise ValueError("Cannot scale silent residual")
    return (wave * min(target_rms/rms(wave), max_peak/peak(wave))).astype(np.float32)

def project_residual(waveform, rms_budget=0.0012, peak_budget=0.0024):
    """Same clip-to-peak THEN RMS-rescale projection as notebook trainer."""
    x = np.clip(np.asarray(waveform, dtype=np.float32), -peak_budget, peak_budget)
    return (x * min(1.0, rms_budget/(rms(x)+1e-12))).astype(np.float32)

def random_residual(rng, n_samples=N_SAMPLES, target_rms=0.0012, peak_budget=0.0024):
    """Uniform matched-RMS noise, as used in the final notebook evaluation."""
    noise = rng.uniform(-1, 1, size=n_samples).astype(np.float32)
    noise *= target_rms / rms(noise)
    if peak(noise) > peak_budget + 1e-7:
        raise ValueError("Noise exceeds peak constraint")
    return noise

def mix(keystroke, carrier, residual=None, clip=True):
    result = np.asarray(keystroke, np.float32) + np.asarray(carrier, np.float32)
    if residual is not None: result = result + np.asarray(residual, np.float32)
    return np.clip(result, -1.0, 1.0) if clip else result
