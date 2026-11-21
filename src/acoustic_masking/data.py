"""Load and split the audio inputs used by the experiment workflows.

Reads labeled letter recordings, creates stratified example-level
splits, validates saved dataset arrays, and builds TensorFlow batches.
Carrier recordings are loaded as mono audio, filtered, and split by time.
"""
from pathlib import Path
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from .config import SAMPLE_RATE, N_SAMPLES, CLASSES

def load_letters(directory):
    directory = Path(directory)
    audio, labels, names = [], [], []
    for label, letter in enumerate(CLASSES):
        for i in range(1,126):
            path = directory / f"{letter}_{i}.wav"
            if not path.exists(): continue  # Notebook behavior; report missing via metadata
            x, sr = librosa.load(path, sr=None, mono=True)
            if sr != SAMPLE_RATE: x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
            x = np.pad(x,(0,max(0,N_SAMPLES-len(x))))[:N_SAMPLES]
            audio.append(x.astype(np.float32)); labels.append(label); names.append(str(path))
    if not audio: raise FileNotFoundError(f"No letter WAVs found in {directory}")
    return np.stack(audio), np.asarray(labels,np.int32), np.asarray(names)

def split_indices(labels, seed=42):
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    train, temp = train_test_split(indices,test_size=0.30,random_state=seed,stratify=labels)
    val, test = train_test_split(temp,test_size=0.50,random_state=seed,stratify=labels[temp])
    return {"train_idx":train,"val_idx":val,"test_idx":test}

def load_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        result = {k:d[k] for k in d.files}
    for k in ("waveforms","labels","train_idx","val_idx","test_idx"):
        if k not in result: raise ValueError(f"Missing key {k} in dataset")
    x,y = result["waveforms"],result["labels"]
    if x.ndim!=2 or x.shape[1]!=N_SAMPLES or len(y)!=len(x): raise ValueError("Dataset shape mismatch")
    splits=[set(result[k].tolist()) for k in ("train_idx","val_idx","test_idx")]
    if any(splits[i]&splits[j] for i,j in ((0,1),(0,2),(1,2))): raise ValueError("Overlapping splits")
    if sum(map(len,splits)) != len(y): raise ValueError("Split coverage mismatch")
    if np.any(y<0) or np.any(y>=len(CLASSES)): raise ValueError("Unexpected class label")
    return result

def batches(x,y,batch_size=32,seed=42,shuffle=False,reshuffle_each_iteration=False):
    import tensorflow as tf
    ds=tf.data.Dataset.from_tensor_slices((np.asarray(x,np.float32),np.asarray(y,np.int32)))
    if shuffle: ds=ds.shuffle(len(y),seed=seed,reshuffle_each_iteration=reshuffle_each_iteration)
    opts=tf.data.Options();opts.experimental_deterministic=True
    return ds.batch(batch_size).with_options(opts).prefetch(tf.data.AUTOTUNE)

def load_carrier_splits(paths):
    """Load mono WAV files, high-pass filter, THEN split chronologically."""
    from .audio import highpass_audio, split_carrier_audio
    return {name:split_carrier_audio(highpass_audio(librosa.load(str(path),sr=SAMPLE_RATE,mono=True)[0]))
            for name,path in paths.items()}
