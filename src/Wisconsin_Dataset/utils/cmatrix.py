# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from sklearn.metrics import confusion_matrix


def pretty_cmatrix(predict, y, method, dtype, filename=False):
    """
    Generates pretty cmatrix plots with matplotlib.pyplot

    Arguments:
    predict -- np.array of values
    y -- np.array of values
    method -- type of detection ["ANN", "SVM"]
    dtype -- type of data ["test", "train"]

    +-----------------+---------------------+---------------------+
    |                 | Predicted Negative  | Predicted Positive  |
    +=================+=====================+=====================+
    | Actual Negative | True Negative (TN)  | False Positive (FP) |
    | Actual Positive | False Negative (FN) | True Positive (TP)  |
    +-----------------+---------------------+---------------------+
    """

    # Special font settings
    font = {"family": "Calibri", "weight": "normal", "size": 30}
    rc("font", **font)

    num_correct = np.sum(y == predict)
    num_samples = len(y)
    percentage_correct = round(num_correct / num_samples * 100, 2)

    cm = np.array(confusion_matrix(y, predict, labels=[0, 1]))
    confusion_df = pd.DataFrame(
        cm,
        index=["Is Cancer", "Is Healthy"],
        columns=["Predicted Cancer", "Predicted Healthy"],
    )

    plt.figure(figsize=(16, 8))
    plt.xticks(rotation="horizontal", fontsize=36)
    plt.yticks(fontsize=36, ha="right", va="center")
    plt.title(
        f"{method} {dtype} Set: {num_samples} Samples\n{num_correct}/{num_samples} {percentage_correct}% Accuracy"
    )
    sns.heatmap(confusion_df, annot=True, fmt="g")

    if filename:
        plt.savefig(filename, dpi=800, transparent=True)

    return confusion_df
