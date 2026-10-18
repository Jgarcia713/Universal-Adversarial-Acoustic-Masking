
"""## Reloading model"""

import json
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

EPSILON = 0.01
STEP_SIZE = 0.001
EPOCHS = 30
# Convert data into TensorFlow datasets
BATCH_SIZE = 32


with np.load("../artifacts/models/keystroke_dataset.npz", allow_pickle=False) as data:
    waveforms = data["waveforms"]
    labels = data["labels"]
    filenames = data["filenames"]

    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

print("\nDataset loaded successfully.")
print("Waveforms:", waveforms.shape)
print("Labels:", labels.shape)
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))



with open("../artifacts/models/preprocessing_config.json") as f:
    config = json.load(f)

model = tf.keras.models.load_model(
    "../artifacts/models/keystroke_cnn.keras",
    compile=False
)
model.trainable = False

with np.load("../artifacts/models/keystroke_dataset.npz", allow_pickle=False) as data:
    waveforms = data["waveforms"]
    labels = data["labels"]
    test_idx = data["test_idx"]

SR = config["sample_rate"]
N_FFT = config["n_fft"]
HOP = config["hop_length"]
N_MELS = config["n_mels"]

# Use librosa's filter-bank coefficients, but convert them to
# a constant TensorFlow tensor for differentiable matrix multiplication.
mel_filters = librosa.filters.mel(
    sr=SR,
    n_fft=N_FFT,
    n_mels=N_MELS
).astype(np.float32)

MEL_FILTERS = tf.constant(mel_filters.T)

# Explicit periodic Hann window, consistent with librosa's STFT.
def hann_window(frame_length, dtype):
    return tf.signal.hann_window(
        frame_length,
        periodic=True,
        dtype=dtype
    )


def tf_power_to_db(mel):
    amin = tf.constant(1e-10, dtype=tf.float32)

    log_spec = 10.0 * (
        tf.math.log(tf.maximum(mel, amin))
        / tf.math.log(tf.constant(10.0, tf.float32))
    )

    # librosa ref=np.max is computed from the
    # original power spectrogram.
    ref_value = tf.reduce_max(
        mel,
        axis=[1, 2],
        keepdims=True
    )

    log_ref = 10.0 * (
        tf.math.log(tf.maximum(ref_value, amin))
        / tf.math.log(tf.constant(10.0, tf.float32))
    )

    log_spec = log_spec - log_ref

    # librosa's default top_db=80
    max_db = tf.reduce_max(
        log_spec,
        axis=[1, 2],
        keepdims=True
    )

    return tf.maximum(log_spec, max_db - 80.0)

def tf_log_mel(audio):
    """
    audio: [batch, samples], float32
    returns: [batch, 64, 50, 1]
    """
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Match librosa STFT center=True and pad_mode="reflect".
    pad = N_FFT // 2
    audio = tf.pad(
        audio,
        [[0, 0], [pad, pad]],
        mode="CONSTANT"
    )

    stft = tf.signal.stft(
        audio,
        frame_length=N_FFT,
        frame_step=HOP,
        fft_length=N_FFT,
        window_fn=hann_window,
        pad_end=False
    )

    power = tf.square(tf.abs(stft))

    # [batch, frames, frequencies]
    #     × [frequencies, mel bands]
    # -> [batch, frames, mel bands]
    mel = tf.matmul(power, MEL_FILTERS)

    # librosa.power_to_db:
    # 10 * log10(max(power, amin) / max(ref, amin))
    amin = tf.constant(1e-10, dtype=tf.float32)

    log_power = 10.0 * (
        tf.math.log(tf.maximum(mel, amin))
        / tf.math.log(tf.constant(10.0, tf.float32))
    )

    # Per-example reference: np.max
    reference = tf.reduce_max(
        log_power,
        axis=[1, 2],
        keepdims=True
    )

    log_mel = log_power - reference

    # Match librosa's top_db=80
    maximum = tf.reduce_max(
        log_mel,
        axis=[1, 2],
        keepdims=True
    )

    log_mel = tf_power_to_db(mel)

    # CNN expects [batch, mel bands, time frames, channels].
    log_mel = tf.transpose(log_mel, [0, 2, 1])
    log_mel = log_mel[..., tf.newaxis]

    return log_mel


"""## Adversary Attack"""

model.trainable = False

train_dataset = tf.data.Dataset.from_tensor_slices((
    waveforms[train_idx].astype(np.float32),
    labels[train_idx].astype(np.int32)
))

train_dataset = (
    train_dataset
    .shuffle(len(train_idx), seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((
    waveforms[test_idx].astype(np.float32),
    labels[test_idx].astype(np.int32)
))

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

N_SAMPLES = waveforms.shape[1]

delta = tf.Variable(
    tf.zeros((N_SAMPLES,), dtype=tf.float32),
    trainable=True
)

print("Perturbation shape:", delta.shape)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False
)

@tf.function
def uap_train_step(audio_batch, label_batch):
    with tf.GradientTape() as tape:
        # Apply the same perturbation to all waveforms
        perturbed = audio_batch + delta[tf.newaxis, :]

        # Keep audio within valid waveform amplitude bounds
        perturbed = tf.clip_by_value(
            perturbed, -1.0, 1.0
        )

        # Differentiable preprocessing
        features = tf_log_mel(perturbed)

        # Frozen classifier
        predictions = model(
            features,
            training=False
        )

        # Maximize classification loss
        loss = loss_fn(label_batch, predictions)

    gradient = tape.gradient(loss, delta)

    # Sign-gradient ascent
    delta.assign_add(
        STEP_SIZE * tf.sign(gradient)
    )

    # Project back into the L-infinity constraint
    delta.assign(
        tf.clip_by_value(
            delta, -EPSILON, EPSILON
        )
    )

    return loss

loss_history = []

for epoch in range(EPOCHS):
    epoch_losses = []

    for audio_batch, label_batch in train_dataset:
        loss = uap_train_step(
            audio_batch,
            label_batch
        )

        epoch_losses.append(float(loss))

    mean_loss = np.mean(epoch_losses)
    loss_history.append(mean_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {mean_loss:.4f}"
    )


def evaluate_perturbation(dataset, perturbation=None):
    total_correct = 0
    total_loss = 0.0
    total_samples = 0

    for audio_batch, label_batch in dataset:
        if perturbation is not None:
            audio_batch = tf.clip_by_value(
                audio_batch + perturbation[tf.newaxis, :],
                -1.0,
                1.0
            )

        features = tf_log_mel(audio_batch)

        predictions = model(
            features,
            training=False
        )

        batch_losses = tf.keras.losses.sparse_categorical_crossentropy(
            label_batch,
            predictions
        )

        predicted_labels = tf.argmax(
            predictions,
            axis=1,
            output_type=tf.int32
        )

        total_correct += int(
            tf.reduce_sum(
                tf.cast(
                    predicted_labels == label_batch,
                    tf.int32
                )
            )
        )

        total_loss += float(
            tf.reduce_sum(batch_losses)
        )

        total_samples += int(tf.shape(audio_batch)[0])

    return {
        "accuracy": total_correct / total_samples,
        "loss": total_loss / total_samples
    }


clean_results = evaluate_perturbation(test_dataset)
uap_results = evaluate_perturbation(test_dataset, delta)

print("Clean:", clean_results)
print("UAP:", uap_results)

rng = np.random.default_rng(42)

random_noise = rng.uniform(
    -EPSILON,
    EPSILON,
    size=N_SAMPLES
).astype(np.float32)

random_noise = tf.constant(random_noise)

random_results = evaluate_perturbation(
    test_dataset,
    random_noise
)

print("\nComparison")
print(f"Clean accuracy:  {clean_results['accuracy']:.2%}")
print(f"Random accuracy: {random_results['accuracy']:.2%}")
print(f"UAP accuracy:    {uap_results['accuracy']:.2%}")


np.save("../artifacts/masks/uap_epsilon_001.npy", delta.numpy())

sf.write(
    "../artifacts/masks/uap_epsilon_001.wav",
    delta.numpy(),
    SR,
    subtype="FLOAT"
)
