import numpy as np
import pytest

def test_paired_reproducible_and_identical_conditions():
    tf=pytest.importorskip('tensorflow')
    from acoustic_masking.config import N_SAMPLES
    from acoustic_masking.evaluation import collect_predictions
    from acoustic_masking.preprocessing import LogMel
    from acoustic_masking.models import build_cnn
    model=build_cnn()
    x=np.random.default_rng(2).normal(0,0.001,(4,N_SAMPLES)).astype(np.float32)
    labels=np.array([0,1,2,3],np.int32)
    source=np.random.default_rng(3).normal(0,0.004,N_SAMPLES*4).astype(np.float32)
    zeros=np.zeros(N_SAMPLES,np.float32)
    kwargs=dict(model=model,log_mel=LogMel(),batches=[(x,labels)],
       carrier_source=source,carrier_rms=0.005,residuals={'zero':zeros},
       seed=42,carrier_name='rain')
    a=collect_predictions(**kwargs)
    b=collect_predictions(**kwargs)
    assert a.equals(b)
    baseline=a[a.condition=='carrier_only'].predicted_label.tolist()
    assert baseline==a[a.condition=='zero'].predicted_label.tolist()
    assert a.carrier_start.nunique()>=1
