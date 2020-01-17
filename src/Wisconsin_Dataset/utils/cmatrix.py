import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def cmatrix(predict, y):
    """
    Print a Confusion Matrix
    TODO: Remove, mostly just a debug tool
    Printing a sklearn cmatrix is likely adequate
    """
    if len(predict) != len(y):
        raise ValueError("Data is inequal!")

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for (i, j) in zip(predict, y):
        if i == 0 and j == 1:
            false_negative += 1
        elif i == 1 and j == 0:
            false_positive += 1
        elif i == 0 and j == 0:
            true_negative += 1
        elif i == 1 and i == 1:
            true_positive += 1

    columns = ["predicted_cancer", "predicted_healthy"]
    rows = ["is_cancer", "is_healthy"]

    return output


def pretty_cmatrix(predict, y, method, dtype):
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

    num_correct = np.sum(y == predict)
    num_samples = len(y)
    percentage_correct = round(num_correct / num_samples * 100, 2)

    cm = np.array(confusion_matrix(y, predict, labels=[0, 1]))
    confusion_df = pd.DataFrame(
        cm,
        index=["is_cancer", "is_healthy"],
        columns=["predicted_cancer", "predicted_healthy"],
    )
    plt.figure(figsize=(8, 4))
    plt.xticks(rotation="horizontal")
    plt.title(
        f"{method} {dtype} Set: {num_samples} Samples\n{num_correct}/{num_samples} {percentage_correct}% Accuracy"
    )
    sns.heatmap(confusion_df, annot=True, fmt="g")

    return confusion_df
