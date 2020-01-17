import pathlib
import time

import cv2
import tensorflow as tf
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


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
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    for i in paths[0]:
        img = load_img(i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for _ in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=pathlib.Path("DATA", "Better", "yes"),
            save_format="jpg",
        ):
            i += 1
            if i > 20:
                break

    for i in paths[1]:
        img = load_img(i)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for _ in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=pathlib.Path("DATA", "Better", "no"),
            save_format="jpg",
        ):
            i += 1
            if i > 20:
                break
