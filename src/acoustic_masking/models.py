"""Build and load the 26-letter keystroke classifier.

Builds the CNN used for supervised classification of log-mel features.
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

def load_frozen(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=False)
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError("Model must have 26 output classes (a-z)")
    model.trainable = False
    return model
