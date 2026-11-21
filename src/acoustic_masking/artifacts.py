"""Read historical residual banks and save completed training runs.

Each saved run has a waveform, per-epoch history, and a JSON manifest
recording the configuration and SHA-256 hashes. Run names identify
individual output directories and cannot be reused.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .audio import rms,peak
from .reproducibility import sha256
from .config import N_SAMPLES,SAMPLE_RATE

def save_run(root, run_name, waveform, history, config, model_path=None, dataset_path=None):
    folder=Path(root)/run_name
    folder.mkdir(parents=True,exist_ok=False)
    wave=np.asarray(waveform,np.float32).reshape(-1)
    if len(wave)!=N_SAMPLES or not np.all(np.isfinite(wave)): raise ValueError("Invalid saved residual")
    path=folder/"residual.npy"; np.save(path,wave)
    history.to_csv(folder/"history.csv",index=False)
    manifest={"run_name":run_name,"created_utc":datetime.now(timezone.utc).isoformat(),
      "sample_rate":SAMPLE_RATE,"n_samples":N_SAMPLES,"rms":rms(wave),"peak":peak(wave),
      "residual_sha256":sha256(path),"config":config,
      "model_sha256":sha256(model_path) if model_path else None,
      "dataset_sha256":sha256(dataset_path) if dataset_path else None}
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return folder

def load_existing_bank(path):
    with np.load(path,allow_pickle=False) as bank:
        return {name:bank[name].astype(np.float32).copy() for name in bank.files}
