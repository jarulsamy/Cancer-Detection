import math
import pathlib
from datetime import datetime

import keras
import scipy
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


def prep_model(num_features):
    """
    Define network architechture
    """
    model = Sequential()
    model.add(Dense(num_features))
    model.add(Activation("relu"))

    for _ in range(3):
        model.add(Dense(512))
        model.add(Activation("relu"))

    model.add(Dropout(1))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]  # or rmsprop
    )

    return model


def train(X, y, epochs=10, batch_size=5, save=False):
    """
    Train a model and save to disk
    """
    num_features = X[0].shape[0]
    model = prep_model(num_features)

    # Tensorboard Logs
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(pathlib.Path(logdir))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    model.fit(
        X,
        y,
        epochs=epochs,
        steps_per_epoch=len(X) // batch_size,
        callbacks=[tensorboard_callback],
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


def binary_step(x):
    if x > 0:
        return 1
    else:
        return 0
