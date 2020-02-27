# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from sklearn.datasets import load_breast_cancer


def load_cancer():
    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(
        np.c_[cancer["data"], cancer["target"]],
        columns=np.append(cancer["feature_names"], ["target"]),
    )

    X = df_cancer.drop(["target"], axis=1)
    y = df_cancer["target"]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=20)

    # Take the first 80% of data as train
    # Remaining 20% is test
    # Avoiding train_test_split() to ensure same data
    # is used for SVM and ANN
    X_train = X[round(len(X) * 0.2) :]
    X_test = X[: round(len(X) * 0.2)]

    y_train = y[round(len(y) * 0.2) :]
    y_test = y[: round(len(y) * 0.2)]

    # Normalize Data
    X_train_min = X_train.min()
    X_train_max = X_train.max()
    X_train_range = X_train_max - X_train_min
    X_train_scaled = (X_train - X_train_min) / (X_train_range)

    X_test_min = X_test.min()
    X_test_range = (X_test - X_test_min).max()
    X_test_scaled = (X_test - X_test_min) / X_test_range

    X_train_scaled = X_train_scaled.values
    X_test_scaled = X_test_scaled.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train_scaled, X_test_scaled, y_train, y_test


def cancer_attributes():
    """
    Returns list of all cancer features.
    """
    cancer = load_breast_cancer()
    return list(cancer["feature_names"])


def correlation_map(save=True):
    # Special font settings
    font = {"family": "Calibri", "weight": "normal", "size": 24}
    rc("font", **font)

    cancer = load_breast_cancer()
    df_cancer = pd.DataFrame(
        np.c_[cancer["data"], cancer["target"]],
        columns=np.append(cancer["feature_names"], ["target"]),
    )

    # plt.figure(figsize=(8, 4))
    plot = sns.pairplot(
        df_cancer,
        hue="target",
        vars=[
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
        ],
    )
    plot.fig.suptitle("Correlation of Tumor Attributes and Malignancy", y=1.08)
    plot.set(xticklabels=[], yticklabels=[])
    plot._legend.remove()

    if save:
        plt.savefig("correlation.png", dpi=800, transparent=True)

    plt.show()


if __name__ == "__main__":
    correlation_map(save=True)
