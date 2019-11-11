# General
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
import time
import json

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
