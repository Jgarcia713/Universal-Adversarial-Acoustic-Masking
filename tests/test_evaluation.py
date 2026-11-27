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

def test_clean_predictions_keep_dataset_indices_and_labels():
    pytest.importorskip('tensorflow')
    from acoustic_masking.config import N_SAMPLES
    from acoustic_masking.evaluation import clean_predictions
    labels=np.array([0,1,2,3,4],np.int32)
    waveforms=np.zeros((5,N_SAMPLES),np.float32)
    def always_b(features,training=False):
        probs=np.zeros((len(features),26),np.float32);probs[:,1]=1.0
        return probs
    frame=clean_predictions(always_b,lambda x:x,waveforms,labels,[4,1,2],'fake',batch_size=2)
    assert frame.dataset_index.tolist()==[4,1,2]
    assert frame.true_label.tolist()==[4,1,2]
    assert frame.predicted_label.tolist()==[1,1,1]
    assert frame.correct.tolist()==[False,True,False]

def test_confusion_matrix_frame_counts_true_rows_and_predicted_columns():
    from acoustic_masking.evaluation import confusion_matrix_frame
    matrix=confusion_matrix_frame([0,0,1,25],[0,1,1,0])
    assert matrix.shape==(26,26) and matrix.to_numpy().sum()==4
    assert matrix.loc['a','a']==1 and matrix.loc['a','b']==1
    assert matrix.loc['b','b']==1 and matrix.loc['z','a']==1

def test_paired_clean_comparison_counts_discordant_pairs():
    from acoustic_masking.evaluation import paired_clean_comparison
    result=paired_clean_comparison([True,True,False,True],[True,False,True,True])
    assert (result['both_correct'],result['a_only_correct'],result['b_only_correct'],result['both_wrong'])==(2,1,1,0)
    assert result['accuracy_difference']==0.0 and result['mcnemar_exact_p']==1.0
    assert paired_clean_comparison([True],[True])['mcnemar_exact_p']==1.0

def _fake_run(correct_by_condition,starts=(10,20)):
    import pandas as pd
    rows=[]
    for repeat in (0,1):
        for condition,correct in correct_by_condition.items():
            for sample_id in (0,1):
                rows.append({'carrier':'rain','repeat':repeat,'sample_id':sample_id,'condition':condition,
                             'correct':correct[sample_id],'carrier_start':starts[sample_id],'residual_phase':5})
    cfg={'seed':1,'playback_scale':0.5,'clip_mixtures':False,
         'carrier_config':{'rain':{'carrier_rms':0.005,'residual_gain':1.0}},
         'residuals':{'multi_scratch':{'bank':'artifacts/masks/residual_bank.npz','key':'multi_scratch'},
                      'zero_101':{'bank':'artifacts/masks/candidate_bank.npz','key':'zero_101'}}}
    return pd.DataFrame(rows),cfg,'sha'

def test_transfer_tables_drop_bank_mean_and_transfer_ratio():
    from acoustic_masking.evaluation import transfer_tables
    reference=_fake_run({'carrier_only':(1,1),'random_noise':(1,1),'multi_scratch':(0,0),'zero_101':(1,0)})
    other=_fake_run({'carrier_only':(1,1),'random_noise':(1,1),'multi_scratch':(1,0),'zero_101':(1,1)})
    by_repeat,summary,comparison=transfer_tables({'cnn':reference,'factored':other})
    row=lambda c:comparison.set_index('condition').loc[c]
    assert row('multi_scratch').drop_cnn==1.0 and row('multi_scratch').drop_factored==0.5
    assert row('multi_scratch').transfer_ratio_factored==0.5
    assert row('bank_mean').accuracy_cnn==0.5 and row('bank_mean').transfer_ratio_factored==0.0
    assert np.isnan(row('carrier_only').transfer_ratio_factored)
    groups=by_repeat.drop_duplicates('condition').set_index('condition').mask_group
    assert groups.to_dict()=={'carrier_only':'control','random_noise':'control','multi_scratch':'historical','zero_101':'bank'}
    assert set(by_repeat.gain[by_repeat.condition!='carrier_only'])=={0.5}
    assert (summary.repeats==2).all()
    ids=by_repeat.drop_duplicates('condition').set_index('condition').mask_id
    assert ids['zero_101']=='artifacts/masks/candidate_bank.npz:zero_101' and ids['carrier_only']=='none'
    assert list(by_repeat.columns[:11])==['model_id','carrier','condition','mask_id','repeat','n_examples',
                                          'n_correct','accuracy','seed','gain','clip_mixtures']

def test_transfer_tables_rejects_unpaired_runs():
    from acoustic_masking.evaluation import transfer_tables
    a=_fake_run({'carrier_only':(1,1),'multi_scratch':(0,0)})
    b=_fake_run({'carrier_only':(1,1),'multi_scratch':(0,0)},starts=(10,99))
    with pytest.raises(ValueError,match='different keystrokes'):
        transfer_tables({'cnn':a,'factored':b})
