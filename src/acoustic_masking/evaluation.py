"""Measure classifier accuracy under paired acoustic masking conditions.

Within a trial, each condition uses the same keystrokes, carrier
windows, and residual phase offsets. Produces prediction-level records,
repeat-level and aggregate accuracy, and prediction concentration metrics.
"""
import numpy as np
import pandas as pd
from .audio import rms, carrier_windows, phase_shift_batch, random_residual
from .config import N_SAMPLES
from .reproducibility import child_seeds


def collect_predictions(model, log_mel, batches, carrier_source, carrier_rms,
                        residuals, seed, carrier_name, repeat=0, clip_mixtures=False):
    """Preserves original paired-eval RNG draw order and -phase indexing.

    clip_mixtures=False matches reported paired notebook results. For new runs,
    clip_mixtures=True matches training's [-1,1] mixing convention. Explicit flag
    prevents silently changing legacy reported results.
    """
    source=np.asarray(carrier_source,np.float32).reshape(-1)
    if len(source)<N_SAMPLES or rms(source)<=0: raise ValueError("Invalid carrier")
    scale=carrier_rms/rms(source)
    validated={}
    for name,waveform in residuals.items():
        wave=np.asarray(waveform,np.float32).reshape(-1)
        if len(wave)!=N_SAMPLES or not np.all(np.isfinite(wave)): raise ValueError(f"Invalid residual {name}")
        validated[name]=wave
    rng=np.random.default_rng(seed)
    rows=[]
    offset=0
    import tensorflow as tf
    for x,y in batches:
        x=np.asarray(x,np.float32)
        if x.ndim==3 and x.shape[-1]==1:x=x[...,0]
        y=np.asarray(y,np.int32).reshape(-1)
        if x.ndim!=2 or x.shape[1]!=N_SAMPLES or len(x)!=len(y): raise ValueError("Invalid keystroke batch")
        bs=len(y)
        starts=rng.integers(0,len(source)-N_SAMPLES+1,size=bs)
        phases=rng.integers(0,N_SAMPLES,size=bs)
        carrier=carrier_windows(source,starts,scale)
        base=x+carrier
        for name in ("carrier_only", *validated.keys()):
            mixture=(base if name=="carrier_only" else base+phase_shift_batch(validated[name],phases))
            if clip_mixtures: mixture=np.clip(mixture,-1.,1.)
            probs=model(log_mel(tf.convert_to_tensor(mixture,tf.float32)),training=False)
            pred=tf.argmax(probs,axis=-1).numpy().reshape(-1)
            for i in range(bs):
                rows.append({"carrier":carrier_name,"repeat":repeat,"sample_id":offset+i,
                  "condition":name,"true_label":int(y[i]),"predicted_label":int(pred[i]),
                  "correct":bool(pred[i]==y[i]),"carrier_start":int(starts[i]),
                  "residual_phase":int(phases[i])})
        offset+=bs
    return pd.DataFrame(rows)


def run_paired_evaluation(model,log_mel,batches,carrier_splits,carrier_config,
                          learned_residuals,repeats=30,seed=20261116,
                          playback_scale=0.90,noise_rms=0.0012,noise_peak=0.0024,
                          split="val",clip_mixtures=False):
    frames=[]
    for carrier_idx,(name,cfg) in enumerate(carrier_config.items()):
        for rep in range(repeats):
            trial_seed,noise_seed=child_seeds(seed,carrier_idx,rep)
            noise=random_residual(np.random.default_rng(noise_seed),target_rms=noise_rms,peak_budget=noise_peak)
            gain=float(cfg["residual_gain"])*playback_scale
            residuals={"random_noise":noise*gain,
                       **{n:np.asarray(w,np.float32)*gain for n,w in learned_residuals.items()}}
            frames.append(collect_predictions(model,log_mel,batches,
                carrier_splits[name][split],float(cfg["carrier_rms"]),residuals,
                trial_seed,name,rep,clip_mixtures=clip_mixtures))
    if not frames: raise ValueError("No evaluation trials")
    predictions=pd.concat(frames,ignore_index=True)
    by_repeat=(predictions.groupby(["carrier","repeat","condition"],sort=False)["correct"].mean()
               .reset_index(name="accuracy"))
    summary=by_repeat.groupby(["carrier","condition"])["accuracy"].agg(["mean","std"]).reset_index()
    return predictions,by_repeat,summary


def prediction_concentration(predictions,target_class=1):
    rows=[]
    for (carrier,condition),g in predictions.groupby(["carrier","condition"]):
        counts=g.predicted_label.value_counts()
        incorrect=g.loc[~g.correct]
        errors=incorrect.predicted_label.value_counts()
        non_target=g[g.true_label!=target_class]
        target=g[g.true_label==target_class]
        rows.append({"carrier":carrier,"condition":condition,"accuracy":float(g.correct.mean()),
             "top_class":int(counts.index[0]),"top_share":float(counts.iloc[0]/len(g)),
             "top_error_class":int(errors.index[0]) if len(errors) else None,
             "top_error_share":float(errors.iloc[0]/len(incorrect)) if len(errors) else float('nan'),
             "unique_classes":len(counts),"target_false_positive_rate":float((non_target.predicted_label==target_class).mean()),
             "target_recall":float((target.predicted_label==target_class).mean()) if len(target) else float('nan')})
    return pd.DataFrame(rows)
