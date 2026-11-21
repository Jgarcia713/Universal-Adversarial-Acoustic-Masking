import numpy as np
import pytest
from acoustic_masking.audio import (rms,peak,split_carrier_audio,normalize_source,
    carrier_windows,phase_shift_batch,scale_to_budget,project_residual,random_residual)
from acoustic_masking.config import N_SAMPLES

def test_carrier_split_and_global_rms():
    x=np.linspace(-1,1,N_SAMPLES*20,dtype=np.float32)
    splits=split_carrier_audio(x)
    assert len(splits['train'])+len(splits['val'])+len(splits['test'])==len(x)
    normalized=normalize_source(splits['train'],0.005)
    assert np.isclose(rms(normalized),0.005,rtol=1e-6)
    windows=carrier_windows(normalized,np.array([0,100]))
    assert windows.shape==(2,N_SAMPLES)
    assert not np.isclose(rms(windows[0]),rms(windows[1]))

def test_shared_phases_and_peak_budget():
    x=np.zeros(N_SAMPLES,np.float32);x[0]=1
    shifted=phase_shift_batch(x,[0,1,5])
    assert np.argmax(shifted,axis=1).tolist()==[0,1,5]
    y=np.random.default_rng(42).normal(size=N_SAMPLES)
    bounded=project_residual(y)
    assert rms(bounded)<=0.0012+1e-8
    assert peak(bounded)<=0.0024+1e-8
    original=scale_to_budget(y)
    assert rms(original)<=0.0012+1e-8
    assert peak(original)<=0.0024+1e-8
    noise=random_residual(np.random.default_rng(5))
    assert np.isclose(rms(noise),0.0012,rtol=1e-5)
    assert peak(noise)<=0.0024+1e-7

def test_split_rejects_too_short_source():
    with pytest.raises(ValueError):split_carrier_audio(np.zeros(N_SAMPLES))
