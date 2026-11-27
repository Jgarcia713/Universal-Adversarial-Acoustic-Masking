import numpy as np
import pytest

def test_factored_cnn_accepts_repo_features_and_outputs_26_probabilities():
    tf=pytest.importorskip('tensorflow')
    from acoustic_masking.models import build_factored_cnn
    from acoustic_masking.preprocessing import LogMel
    from acoustic_masking.config import N_SAMPLES
    audio=np.random.default_rng(7).normal(0,0.02,size=(3,N_SAMPLES)).astype(np.float32)
    features=LogMel()(audio)
    model=build_factored_cnn()
    out=model(features,training=False).numpy()
    assert features.shape==(3,64,50,1) and model.input_shape==(None,64,50,1)
    assert out.shape==(3,26)
    assert np.allclose(out.sum(axis=1),1.0,atol=1e-5)

def test_factored_cnn_uses_factored_kernels_unlike_baseline():
    tf=pytest.importorskip('tensorflow')
    from acoustic_masking.models import build_cnn,build_factored_cnn
    def kernels(model):
        return {layer.kernel_size for layer in model.layers if isinstance(layer,tf.keras.layers.Conv2D)}
    assert kernels(build_factored_cnn())=={(3,3),(3,1),(1,3)}
    assert kernels(build_cnn())=={(3,3)}

def test_factored_cnn_saves_and_loads_through_evaluation_loader(tmp_path):
    tf=pytest.importorskip('tensorflow')
    from acoustic_masking.models import build_factored_cnn,load_frozen
    model=build_factored_cnn()
    x=np.random.default_rng(3).normal(size=(2,64,50,1)).astype(np.float32)
    path=tmp_path/'factored.keras'
    model.save(path)
    loaded=load_frozen(path)
    assert not loaded.trainable
    assert np.allclose(loaded(x,training=False).numpy(),model(x,training=False).numpy(),atol=1e-6)
