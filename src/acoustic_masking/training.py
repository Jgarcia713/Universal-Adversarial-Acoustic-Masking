"""Optimize masking waveforms against a frozen keystroke classifier.

The carrier-aware trainer uses Adam to maximize classification loss over
sampled mixtures and circular residual shifts, subject to peak and RMS
budgets. The original sign-gradient baseline is available separately.
"""
import numpy as np
import pandas as pd
from .config import N_SAMPLES
from .audio import rms, peak, normalize_source


def train_residual(model, log_mel, carrier_sources, carrier_config, train_dataset,
                   initial_residual=None, rms_budget=0.0012, peak_budget=0.0024,
                   epochs=320, steps_per_epoch=12, learning_rate=1e-5,
                   seed=42, init_mode="zero", init_std=1e-5, verbose=True,
                   compile_step=False, on_epoch_end=None):
    """Frozen-model cross-entropy ascent; same projections and +shift convention as notebook.

    Empty carrier_config/source mappings mean keystroke-only UAP training; no dummy carrier.
    For nonzero carrier_rms, source normalization uses RMS of full train split.
    Pass a freshly built deterministic train_dataset per run for replayability.
    """
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    if set(carrier_sources) != set(carrier_config):
        raise ValueError("Carrier source and configuration names differ")
    names = list(carrier_config)
    prepared={}
    for name in names:
        c=carrier_config[name]
        if c["residual_gain"] < 0: raise ValueError("Residual gain must be nonnegative")
        prepared[name]=normalize_source(carrier_sources[name], float(c["carrier_rms"]))
    if initial_residual is not None:
        initial=np.asarray(initial_residual,np.float32).reshape(-1).copy()
    elif init_mode == "zero": initial=np.zeros(N_SAMPLES,np.float32)
    elif init_mode == "normal": initial=rng.normal(0, init_std, N_SAMPLES).astype(np.float32)
    else: raise ValueError("init_mode must be 'zero' or 'normal'")
    if len(initial)!=N_SAMPLES: raise ValueError("Initial residual length incorrect")
    delta=tf.Variable(initial,dtype=tf.float32)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    def project():
        delta.assign(tf.clip_by_value(delta,-peak_budget,peak_budget))
        r=tf.sqrt(tf.reduce_mean(tf.square(delta)))
        delta.assign(delta*tf.minimum(1.0,rms_budget/(r+1e-12)))
    project()
    model.trainable=False

    def step(audio, labels, carrier, gains, gather_idx):
        with tf.GradientTape() as tape:
            windows = tf.gather(delta, gather_idx) * gains
            mixed = tf.clip_by_value(audio + carrier + windows, -1., 1.)
            probabilities = model(log_mel(mixed), training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))
        gradient = tape.gradient(loss, delta)
        if gradient is None:
            raise RuntimeError("Missing residual gradient")
        tf.debugging.assert_all_finite(gradient, "Invalid residual gradient")
        optimizer.apply_gradients([(-gradient, delta)])
        project()
        return loss

    if compile_step:
        step = tf.function(step, reduce_retracing=True)
    history=[]
    indices=np.arange(N_SAMPLES)
    if steps_per_epoch <= 0 or epochs <= 0: raise ValueError("epochs and steps_per_epoch must be positive")
    for epoch in range(epochs):
        losses=[]
        for audio_batch, label_batch in train_dataset.take(steps_per_epoch):
            audio_batch=tf.cast(audio_batch,tf.float32)
            label_batch=tf.reshape(tf.cast(label_batch,tf.int32),[-1])
            bs=int(tf.shape(audio_batch)[0])
            if names:
                ids=np.tile(np.arange(len(names)),int(np.ceil(bs/len(names))))[:bs]
                rng.shuffle(ids)
                carrier=np.zeros((bs,N_SAMPLES),np.float32)
                gains=np.empty(bs,np.float32)
                for carrier_id,name in enumerate(names):
                    positions=np.where(ids==carrier_id)[0]
                    if not len(positions): continue
                    source=prepared[name]
                    starts=rng.integers(0,len(source)-N_SAMPLES+1,size=len(positions))
                    carrier[positions]=source[starts[:,None]+indices[None,:]]
                    gains[positions]=float(carrier_config[name]["residual_gain"])
            else:
                carrier=np.zeros((bs,N_SAMPLES),np.float32)
                gains=np.ones(bs,np.float32)
            shifts=rng.integers(0,N_SAMPLES,size=bs)
            # Match original training: delta[(i + shift) % N]; uniform random shifts.
            gather_idx=tf.convert_to_tensor((indices[None,:]+shifts[:,None])%N_SAMPLES,tf.int32)
            loss = step(audio_batch, label_batch, tf.convert_to_tensor(carrier),
                        tf.convert_to_tensor(gains[:, None]), gather_idx)
            losses.append(float(loss.numpy()))
        if not losses: raise ValueError("Training dataset yielded no batches")
        wave=delta.numpy()
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"rms":rms(wave),"peak":peak(wave)})
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, wave)
        if verbose and ((epoch+1)%10==0 or epoch==0):
            print(f"Epoch {epoch+1}/{epochs} loss={history[-1]['loss']:.4f} RMS={rms(wave):.6f} peak={peak(wave):.6f}")
    return delta.numpy().copy(),pd.DataFrame(history)


def train_original_uap(model, log_mel, train_dataset, epsilon=0.01,
                       step_size=0.001, epochs=30):
    """Historical sign-gradient UAP baseline (not phase-robust)."""
    import tensorflow as tf
    model.trainable=False
    delta=tf.Variable(tf.zeros(N_SAMPLES,dtype=tf.float32))
    history=[]
    for epoch in range(epochs):
        losses=[]
        for x,y in train_dataset:
            with tf.GradientTape() as tape:
                mixture=tf.clip_by_value(tf.cast(x,tf.float32)+delta[None,:],-1.,1.)
                probs=model(log_mel(mixture),training=False)
                loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y,probs))
            grad=tape.gradient(loss,delta)
            if grad is None: raise RuntimeError("No gradient")
            delta.assign_add(step_size*tf.sign(grad))
            delta.assign(tf.clip_by_value(delta,-epsilon,epsilon))
            losses.append(float(loss.numpy()))
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses)),"rms":rms(delta.numpy()),"peak":peak(delta.numpy())})
    return delta.numpy().copy(),pd.DataFrame(history)
