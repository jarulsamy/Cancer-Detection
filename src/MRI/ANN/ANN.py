# -*- coding: utf-8 -*-
import pathlib
import random
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from utils import loader

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

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(train, test, save=False):
    """
    Train a model and optionally save to disk
    """
    model = prep_model()
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(pathlib.Path(logdir))

    # Tensorboard Logs
    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

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
        callbacks=[tensorboard_callback],
    )

    if save:
        # Serialize to json and save
        model_json = model.to_json()
        with open("model.json", "w") as f:
            f.write(model_json)
        model.save_weights("model.h5")

    return model


def step(x, s=0.5):
    if x >= s:
        return 1
    else:
        return 0


def post_train_examples(files, model, thresh=0.8, dset=None):
    """
    Show some example images with classifications.
    """
    # Load some demo images
    files = loader.load_data(files)
    yes = [cv2.imread(random.choice(files[0])) for _ in range(2)]
    no = [cv2.imread(random.choice(files[1])) for _ in range(2)]

    # Resize
    yes = [cv2.resize(img, (img_height, img_width)) for img in yes]
    no = [cv2.resize(img, (img_height, img_width)) for img in no]

    images = {"Cancerous": yes, "Healthy": no}

    index = 1
    plt.figure(figsize=[6, 6])
    plt.title(f"ANN {dset} Set")
    for (k, v) in images.items():
        for i in v:
            plt.subplot(2, 2, index)
            data = tf.cast(i, tf.float32)
            pred = model.predict(data[np.newaxis, ...])
            pred = "Cancerous" if pred == 1 else "Healthy"
            plt.title(f"Pred: {pred}, Actual {k}")
            plt.imshow(i, cmap="gray")
            index += 1

    yes = files[0]
    no = files[1]

    yes = [cv2.resize(cv2.imread(img), (img_height, img_width)) for img in yes]
    no = [cv2.resize(cv2.imread(img), (img_height, img_width)) for img in no]

    X = yes + no
    X = [tf.cast(i, tf.float32)[np.newaxis, ...] for i in X]
    predict = np.asarray([model.predict(i) for i in X])
    predict = predict.flatten()
    predict = [step(i, thresh) for i in predict]
    y = np.asarray([1 for _ in yes] + [0 for _ in no])

    num_correct = np.sum(y == predict)
    num_samples = len(y)
    percentage_correct = round(num_correct / num_samples * 100, 2)

    cm = np.array(confusion_matrix(y, predict, labels=[0, 1]))
    confusion_df = pd.DataFrame(
        cm,
        index=["is_cancer", "is_healthy"],
        columns=["predicted_cancer", "predicted_healthy"],
    )

    plt.figure(figsize=[8, 4])
    plt.xticks(rotation="horizontal")
    plt.title(
        f"ANN {dset} Set: {num_samples} Samples\n{num_correct}/{num_samples} {percentage_correct}% Accuracy"
    )
    sns.heatmap(confusion_df, annot=True, fmt="g")

    plt.show()
