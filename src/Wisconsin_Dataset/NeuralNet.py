# -*- coding: utf-8 -*-
import math
import os
import pathlib
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Suppress tensorflow verbose output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


def prep_model(num_features):
    """
    Define network architechture
    """
    model = Sequential()
    model.add(Dense(512, input_dim=num_features))
    model.add(Activation("sigmoid"))

    for _ in range(3):
        model.add(Dense(512))
        model.add(Activation("relu"))

    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


def train(X, y, epochs=10, batch_size=5, save=False):
    """
    Train a model and, optionally, save to disk
    """
    num_features = X[0].shape[0]
    model = prep_model(num_features)

    # Tensorboard Logs
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(pathlib.Path(logdir))
    tensorboard_callback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    model.fit(
        X,
        y,
        epochs=epochs,
        steps_per_epoch=len(X) // batch_size,
        callbacks=[tensorboard_callback],
        validation_split=0.2,
    )

    if save:
        model_filename = "model-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
        model_json = model.to_json()
        with open(model_filename, "w") as f:
            f.write(model_json)
        model.save_weights("model-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5")

    return model


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def step(x, s=0.5):
    if x >= s:
        return 1
    else:
        return 0
