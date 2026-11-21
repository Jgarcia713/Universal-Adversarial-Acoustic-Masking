"""Shared signal dimensions, class labels, and experiment settings.

Defines the sample rate, window length, log-mel parameters, alphabet
ordering, and default carrier settings. Loads YAML experiment files
and checks that their top-level value is a mapping.
"""
from pathlib import Path
import yaml

SAMPLE_RATE = 44100
N_SAMPLES = 11025
N_FFT = 1024
HOP_LENGTH = 225
N_MELS = 64
CLASSES = list("abcdefghijklmnopqrstuvwxyz")
DEFAULT_CARRIERS = {
    "rain": {"carrier_rms": 0.005, "residual_gain": 1.0},
    "fan": {"carrier_rms": 0.010, "residual_gain": 0.75},
    "wind": {"carrier_rms": 0.010, "residual_gain": 1.0},
}

def read_config(path):
    with Path(path).open(encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return result
