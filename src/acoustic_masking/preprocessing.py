"""Convert keystroke waveform windows into classifier-ready log-mel features.

Provides the librosa reference used for classifier training and a
numerically matched TensorFlow implementation. TensorFlow keeps the
features differentiable with respect to the waveform for residual training.
"""
import librosa
import numpy as np
from .config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

def librosa_log_mel(audio):
    mel = librosa.feature.melspectrogram(y=np.asarray(audio, dtype=np.float32), sr=SAMPLE_RATE,
                n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, pad_mode="constant")
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

class LogMel:
    """Caches mel filters; use the same instance throughout an experiment."""
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        self.mel_filters = tf.constant(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT,
                             n_mels=N_MELS).astype(np.float32).T)

    def __call__(self, audio):
        tf = self.tf
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        audio = tf.pad(audio, [[0,0], [N_FFT//2,N_FFT//2]], mode="CONSTANT")
        stft = tf.signal.stft(audio, frame_length=N_FFT, frame_step=HOP_LENGTH,
            fft_length=N_FFT, window_fn=lambda size,dtype: tf.signal.hann_window(size, periodic=True,dtype=dtype))
        mel = tf.matmul(tf.square(tf.abs(stft)), self.mel_filters)
        amin = tf.constant(1e-10, tf.float32)
        log_power = 10.0 * tf.math.log(tf.maximum(mel, amin))/tf.math.log(tf.constant(10.0,tf.float32))
        reference = tf.reduce_max(log_power,axis=[1,2],keepdims=True)
        log_mel = log_power-reference
        max_db = tf.reduce_max(log_mel,axis=[1,2],keepdims=True)
        log_mel = tf.maximum(log_mel, max_db-80.0)
        return tf.transpose(log_mel,[0,2,1])[...,tf.newaxis]
