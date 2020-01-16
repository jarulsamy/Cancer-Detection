import json
import os
import pathlib
import random
import time
from datetime import datetime

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.regularizers import l2
from keras.utils import plot_model
from PIL import Image

from . import loader

# Size to resize images to
img_width, img_height = 150, 150
EPOCHS = 15
batch_size = 16


def prep_model():
    """
    Define network architechture
    """
    # Build appropiate model shape
    if K.image_data_format() == "channels_first":
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]  # or adam
    )

    return model


# Callback class to log metrics to tensorboard
class Metrics(keras.callbacks.Callback):
    def __init__(self, logdir):
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logdir = str(pathlib.Path(logdir))

    def on_train_begin(self, logs=None):
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        self.accuracy = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.accuracy.append(logs.get("accuracy"))
        self.losses.append(logs.get("loss"))

    # Sometimes duplicates, seems like a tensorboard bug.
    def on_epoch_end(self, epoch, logs):
        with self.summary_writer.as_default():
            tf.summary.scalar("Accuracy", logs.get("accuracy"), step=epoch)
            tf.summary.scalar("Loss", logs.get("loss"), step=epoch)


def train(train, test):
    """
    Train a model and save to disk
    """
    model = prep_model()

    # Tensorboard Logs

    metrics = Metrics()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

    # Augmentation for training set, many changes
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    # Augmentation for testing set, only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
    )

    validation_generator = test_datagen.flow_from_directory(
        test,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[tensorboard_callback, metrics],
    )

    # Serialize to json and save
    model_json = model.to_json()
    with open("model.json", "w") as f:
        f.write(model_json)
    model.save_weights("model.h5")

    return model


def post_train_examples(files, model):
    """
    Show some example images with classifications.
    """
    # Load some demo images
    files = loader.load_data(files)
    yes = [cv2.imread(random.choice(files[0])) for _ in range(2)]
    no = [cv2.imread(random.choice(files[1])) for _ in range(2)]

    # Resize
    yes = [cv2.resize(i, (img_height, img_width)) for i in yes]
    no = [cv2.resize(i, (img_height, img_width)) for i in no]

    images = {"Yes": yes, "No": no}

    index = 1
    plt.figure(figsize=[6, 6])
    for i in images:
        for j in images[i]:
            plt.subplot(2, 2, index)
            pred = model.predict(j[np.newaxis, ...])
            pred = "Yes" if pred == 1 else "No"
            plt.title(f"Pred: {pred}, Actual: {i}")
            plt.imshow(j, cmap="gray")
            index += 1

    plt.show()
