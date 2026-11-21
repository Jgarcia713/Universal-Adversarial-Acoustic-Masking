import numpy as np
import pytest

def test_librosa_tensorflow_parity_and_gradient():
    tf=pytest.importorskip('tensorflow')
    from acoustic_masking.preprocessing import librosa_log_mel,LogMel
    from acoustic_masking.config import N_SAMPLES
    audio=np.random.default_rng(21).normal(0,0.02,size=(3,N_SAMPLES)).astype(np.float32)
    reference=np.stack([librosa_log_mel(x) for x in audio])
    variable=tf.Variable(audio)
    features=LogMel()
    with tf.GradientTape() as tape:
        actual=features(variable)[...,0]
        objective=tf.reduce_sum(actual)
    gradient=tape.gradient(objective,variable)
    assert actual.shape==(3,64,50)
    assert np.allclose(reference,actual.numpy(),atol=1e-3,rtol=1e-4)
    assert gradient is not None and np.isfinite(gradient.numpy()).all()
