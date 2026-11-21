import numpy as np
from acoustic_masking.data import split_indices

def test_stratified_split_deterministic_and_disjoint():
    labels=np.repeat(np.arange(26),20)
    a=split_indices(labels,seed=42)
    b=split_indices(labels,seed=42)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    indices=[set(a[k]) for k in ('train_idx','val_idx','test_idx')]
    assert not(indices[0]&indices[1] or indices[0]&indices[2] or indices[1]&indices[2])
    assert sum(map(len,indices))==len(labels)
