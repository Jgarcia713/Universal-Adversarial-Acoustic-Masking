"""Measure classifier accuracy under paired acoustic masking conditions.

Within a trial, each condition uses the same keystrokes, carrier
windows, and residual phase offsets. Produces prediction-level records,
repeat-level and aggregate accuracy, and prediction concentration metrics.
"""
import numpy as np
import pandas as pd
from .audio import rms, carrier_windows, phase_shift_batch, random_residual
from .config import N_SAMPLES, CLASSES
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


def clean_predictions(model, log_mel, waveforms, labels, indices, model_id,
                      filenames=None, batch_size=64):
    """Predict unmasked keystrokes. One row per example, keyed by dataset index."""
    import tensorflow as tf
    indices=np.asarray(indices)
    rows=[]
    for start in range(0,len(indices),batch_size):
        idx=indices[start:start+batch_size]
        probs=np.asarray(model(log_mel(tf.convert_to_tensor(waveforms[idx],tf.float32)),training=False))
        if probs.ndim!=2 or probs.shape[1]!=len(CLASSES): raise ValueError("Model must output [B,26]")
        pred=probs.argmax(axis=1)
        for j,i in enumerate(idx):
            rows.append({"model_id":model_id,"dataset_index":int(i),
                         "filename":"" if filenames is None else str(filenames[i]),
                         "true_label":int(labels[i]),"predicted_label":int(pred[j]),
                         "correct":bool(pred[j]==labels[i]),"confidence":float(probs[j,pred[j]])})
    return pd.DataFrame(rows)


def confusion_matrix_frame(true_labels, predicted_labels):
    """26x26 counts; rows are true letters, columns are predicted letters."""
    matrix=np.zeros((len(CLASSES),len(CLASSES)),dtype=int)
    np.add.at(matrix,(np.asarray(true_labels,int),np.asarray(predicted_labels,int)),1)
    return pd.DataFrame(matrix,index=pd.Index(CLASSES,name="true"),
                        columns=pd.Index(CLASSES,name="predicted"))


def paired_clean_comparison(correct_a, correct_b):
    """Compare two models on the same examples using discordant pairs.

    The exact McNemar p-value is a two-sided binomial test on the examples
    exactly one model got right.
    """
    from scipy.stats import binomtest
    a=np.asarray(correct_a,bool);b=np.asarray(correct_b,bool)
    if a.shape!=b.shape or a.ndim!=1: raise ValueError("Paired results must be equal-length 1-D arrays")
    a_only=int((a&~b).sum());b_only=int((~a&b).sum());discordant=a_only+b_only
    return {"n_examples":int(len(a)),"both_correct":int((a&b).sum()),
            "a_only_correct":a_only,"b_only_correct":b_only,"both_wrong":int((~a&~b).sum()),
            "accuracy_difference":float(a.mean()-b.mean()),
            "mcnemar_exact_p":float(binomtest(a_only,discordant,0.5).pvalue) if discordant else 1.0}


def transfer_tables(runs, reference=None):
    """Combine paired masking evaluations of several classifiers.

    runs maps model_id -> (predictions DataFrame, evaluation config, model sha256).
    Every run must share identical keystrokes, carrier windows and phases.

    Returns three DataFrames. The first has one row per model, carrier,
    condition and repeat. The second summarises across repeats. The third puts
    the models side by side, where transfer_ratio is a model's mean accuracy
    drop (relative to carrier_only in the same repeat) divided by the reference
    model's drop. mask_id names the mask file, as "bank.npz:key" or a .npy path.
    """
    model_ids=list(runs);reference=reference or model_ids[0]
    draw_keys=["carrier","repeat","sample_id","carrier_start","residual_phase"]
    reference_draws=None;frames=[]
    for model_id,(pred,cfg,model_sha) in runs.items():
        draws=(pred.loc[pred.condition=="carrier_only",draw_keys]
               .sort_values(draw_keys[:3]).reset_index(drop=True))
        if reference_draws is None: reference_draws=draws
        elif not draws.equals(reference_draws):
            raise ValueError(f"{model_id} used different keystrokes, carrier windows or phases")
        groups={"carrier_only":"control","random_noise":"control",
                **{n:"bank" if "candidate_bank" in str(e.get("bank","")) else "historical"
                   for n,e in cfg["residuals"].items()}}
        mask_ids={"carrier_only":"none","random_noise":"generated:random_residual",
                  **{n:f'{e["bank"]}:{e["key"]}' if "bank" in e else str(e.get("path",""))
                     for n,e in cfg["residuals"].items()}}
        playback=float(cfg.get("playback_scale",0.90))
        gains={c:float(v["residual_gain"])*playback for c,v in cfg["carrier_config"].items()}
        counts=(pred.groupby(["carrier","condition","repeat"],sort=False)["correct"]
                .agg(n_examples="size",n_correct="sum").reset_index())
        counts.insert(0,"model_id",model_id)
        counts["mask_group"]=counts.condition.map(groups)
        counts["mask_id"]=counts.condition.map(mask_ids)
        counts["accuracy"]=counts.n_correct/counts.n_examples
        counts["gain"]=[0.0 if cond=="carrier_only" else gains[c] for c,cond in zip(counts.carrier,counts.condition)]
        counts["seed"]=int(cfg["seed"]);counts["clip_mixtures"]=bool(cfg.get("clip_mixtures",False))
        counts["model_sha256"]=model_sha
        frames.append(counts)
    by_repeat=pd.concat(frames,ignore_index=True)
    baseline=by_repeat[by_repeat.condition=="carrier_only"].set_index(["model_id","carrier","repeat"]).accuracy
    keys=pd.MultiIndex.from_frame(by_repeat[["model_id","carrier","repeat"]])
    by_repeat["drop_vs_carrier_only"]=baseline.reindex(keys).to_numpy()-by_repeat.accuracy.to_numpy()
    bank=(by_repeat[by_repeat.mask_group=="bank"].groupby(["model_id","carrier","repeat"])
          [["accuracy","drop_vs_carrier_only"]].mean().reset_index()
          .assign(condition="bank_mean",mask_group="bank_mean"))
    rows=pd.concat([by_repeat[bank.columns],bank],ignore_index=True)
    summary=(rows.groupby(["model_id","carrier","mask_group","condition"])
             .agg(repeats=("accuracy","size"),accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),
                  drop_mean=("drop_vs_carrier_only","mean"),drop_std=("drop_vs_carrier_only","std"))
             .reset_index())
    wide=summary.pivot(index=["carrier","mask_group","condition"],columns="model_id",
                       values=["accuracy_mean","drop_mean"])
    comparison=pd.DataFrame(index=wide.index)
    for model_id in model_ids:
        comparison[f"accuracy_{model_id}"]=wide[("accuracy_mean",model_id)]
        comparison[f"drop_{model_id}"]=wide[("drop_mean",model_id)]
    for model_id in model_ids:
        if model_id==reference: continue
        ref_drop=comparison[f"drop_{reference}"]
        comparison[f"transfer_ratio_{model_id}"]=(comparison[f"drop_{model_id}"]/ref_drop).where(ref_drop>0)
    # Handoff column order from docs/01_STRONGER_CLASSIFIER_TEAMMATE.md, then the extras.
    lead=["model_id","carrier","condition","mask_id","repeat","n_examples","n_correct",
          "accuracy","seed","gain","clip_mixtures"]
    by_repeat=by_repeat[lead+[c for c in by_repeat.columns if c not in lead]]
    return by_repeat,summary,comparison.reset_index()
