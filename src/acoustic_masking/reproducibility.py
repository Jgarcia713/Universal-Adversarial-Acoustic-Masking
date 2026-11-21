"""Seed experiment randomness and record the environment used for a run.

Sets NumPy, Python, and TensorFlow seeds, derives child seeds for
evaluation trials, computes SHA-256 file hashes, and records software
versions and visible GPUs alongside the results.
"""
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
import numpy as np

def set_seed(seed, deterministic=True):
    # PYTHONHASHSEED must be set before Python starts to affect hashing.
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()

def child_seeds(seed, *identifiers, count=2):
    return tuple(int(x) for x in np.random.SeedSequence([seed, *identifiers]).generate_state(count))

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def environment_info():
    import librosa, scipy, pandas, sklearn, tensorflow as tf
    return {
        "python": sys.version, "platform": platform.platform(),
        "tensorflow": tf.__version__, "numpy": np.__version__,
        "librosa": librosa.__version__, "scipy": scipy.__version__,
        "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED")
    }

def save_environment(path):
    Path(path).write_text(json.dumps(environment_info(), indent=2), encoding="utf-8")
