from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    TRIAL_SHAPE, CNN_FILTERS, CNN_KERNEL_SIZE,
    CNN_DROPOUT, CNN_L2, CNN_DENSE_UNITS, CNN_LR, N_CLASSES,
)


def build_cnn(
    input_shape: tuple = TRIAL_SHAPE,
    n_classes:   int   = N_CLASSES,
) -> Model:
    reg = regularizers.l2(CNN_L2)
    inp = layers.Input(shape=input_shape, name="cwt_image")

    x = layers.Conv2D(
        CNN_FILTERS, CNN_KERNEL_SIZE,
        padding="valid", activation="relu",
        kernel_regularizer=reg,
        name="conv1",
    )(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop1")(x)

    x = layers.Conv2D(
        CNN_FILTERS, CNN_KERNEL_SIZE,
        padding="valid", activation="relu",
        kernel_regularizer=reg,
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(CNN_DROPOUT, name="drop2")(x)

    x   = layers.Flatten(name="flatten")(x)
    x   = layers.Dense(CNN_DENSE_UNITS, activation="relu",
                       kernel_regularizer=reg, name="fc1")(x)
    out = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="cnn_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=CNN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
