# -*- coding: utf-8 -*-
import keras
import matplotlib.pyplot as plt
import numpy as np


class TrainingPlot(keras.callbacks.Callback):

    # Called when the training begins
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # Called at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get("loss"))
        self.acc.append(logs.get("acc"))
        self.val_losses.append(logs.get("val_loss"))
        self.val_acc.append(logs.get("val_acc"))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig("output/Epoch-{}.png".format(epoch))
            plt.close()
