"""Build and load the 26-letter keystroke classifier.

Builds the CNN used for supervised classification of log-mel features, plus
the factored-convolution CNN from the upstream notebook as a second model.
Loading a saved Keras model validates the number of output classes and
freezes its weights for residual training and evaluation.
"""
from .config import CLASSES, N_MELS

def build_cnn():
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        keras.Input(shape=(64, 50, 1)),
        layers.Conv2D(32,(3,3),activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.2),
        layers.Conv2D(128,(3,3),activation="relu"), layers.BatchNormalization(), layers.MaxPooling2D((2,2)), layers.Dropout(0.3),
        layers.Flatten(), layers.Dense(128,activation="relu"), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(len(CLASSES),activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

def build_factored_cnn(filters=(32, 64), dense_units=256, dropout_conv=0.2,
                       dropout_dense=0.4, learning_rate=1e-3):
    """Factored-convolution CNN from notebooks/ZnanstveniRad_final.ipynb (cell 14).

    Defaults are the notebook's final "paper config 13" (cell 19). Changes from
    the notebook: 26 letter classes, sparse integer labels instead of one-hot,
    and inputs from this repo's log-mel transform.

    Input is [64 mel bins, 50 frames, 1], so MaxPooling2D((2, 1)) halves the
    mel (frequency) axis and keeps every time frame.
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    f1, f2 = filters
    model = keras.Sequential([
        keras.Input(shape=(N_MELS, 50, 1)),
        # Block 1: 3x3, then 3x1 and 1x3 factored convolutions, no bias.
        layers.Conv2D(f1,(3,3),activation="relu",padding="same",use_bias=False),
        layers.Conv2D(f1,(3,1),activation="relu",padding="same",use_bias=False),
        layers.Conv2D(f1,(1,3),activation="relu",padding="same",use_bias=False),
        layers.MaxPooling2D((2,1)),
        # Block 2: same pattern with more channels.
        layers.Conv2D(f2,(3,3),activation="relu",padding="same"),
        layers.Conv2D(f2,(3,1),activation="relu",padding="same"),
        layers.Conv2D(f2,(1,3),activation="relu",padding="same"),
        layers.MaxPooling2D((2,1)),
        layers.Dropout(dropout_conv), layers.Flatten(),
        layers.Dense(dense_units,activation="relu"), layers.BatchNormalization(), layers.Dropout(dropout_dense),
        layers.Dense(len(CLASSES),activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

ARCHITECTURES = {"cnn": build_cnn, "factored": build_factored_cnn}

def load_frozen(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=False)
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError("Model must have 26 output classes (a-z)")
    model.trainable = False
    return model
