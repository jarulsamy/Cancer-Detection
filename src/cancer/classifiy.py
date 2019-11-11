# General
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
import psutil
import itertools
import time

# ML
import tensorflow as tf
import scipy
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model


DATA = "Data/"
ORIGINAL = "Original/"
Better = "Better/"
img_width, img_height = 150, 150
EPOCHS = 10
batch_size = 16

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(pathlib.Path(logdir))


def load_data(path: str) -> tuple:
    """
    Grabs paths of all data images
    """

    # Subdirectories for cancerous and non cancerous
    YES = "yes"
    NO = "no"

    yes_path = pathlib.Path(path, YES)
    no_path = pathlib.Path(path, NO)

    yes_files = list()
    no_files = list()

    for (dirpath, dirnames, filenames) in os.walk(yes_path):
        yes_files += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, dirnames, filenames) in os.walk(no_path):
        no_files += [os.path.join(dirpath, file) for file in filenames]

    return (yes_files, no_files)


def generate_better_data(paths: tuple):
    """
    Debug tool to generate tons of augmented images.
    """
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    for i in paths[0]:
        img = load_img(i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=pathlib.Path(DATA, Better, "yes"), save_format="jpg"):
            i += 1
            if i > 20:
                break
    for i in paths[1]:
        img = load_img(i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=pathlib.Path(DATA, Better, "no"), save_format="jpg"):
            i += 1
            if i > 20:
                break


def show_data(paths: tuple):
    """
    Show all the images
    param1: tuple(list, list) # (List_of_yes_images, List_of_no_images)
    """
    for i in paths[0]:
        img = cv2.imread(i)
        cv2.imshow("Yes", img)
        time.sleep(0.01)
        # Quit on ESC
        if cv2.waitKey(1) == 27:
            exit(0)
    for i in paths[1]:
        img = cv2.imread(i)
        cv2.imshow("No", img)
        time.sleep(0.01)
        # Quit on ESC
        if cv2.waitKey(1) == 27:
            exit(0)


def prep_model():
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.summary_writer = tf.summary.create_file_writer(logdir)
        self.accuracy = []
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.accuracy.append(logs.get("accuracy"))
        self.losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            tf.summary.scalar("Accuracy", logs.get("accuracy"), step=epoch)
            tf.summary.scalar("Loss", logs.get("loss"), step=epoch)


def train():
    model = prep_model()

    metrics = Metrics()
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        pathlib.Path(DATA, ORIGINAL),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        # verbose=0,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback, metrics]
    )
    # Always save weights after training or during training
    model.save_weights('first_try.h5')
    return model


def post_train_examples(model=None):
    files = load_data(pathlib.Path(DATA, ORIGINAL))

    # Load some demo images
    yes = [cv2.imread(i) for i in files[0][:2]]
    no = [cv2.imread(i) for i in files[1][:2]]

    # Resize
    yes = [cv2.resize(i, (img_height, img_width)) for i in yes]
    no = [cv2.resize(i, (img_height, img_width)) for i in no]

    images = {
        "Yes": yes,
        "No": no
    }

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


if __name__ == "__main__":
    model = train()
    post_train_examples(model)
