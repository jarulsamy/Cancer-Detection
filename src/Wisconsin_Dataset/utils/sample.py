# -*- coding: utf-8 -*-
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from sklearn.datasets import load_breast_cancer


def _clean_nums(*nums: float, digits=2):
    """
    Round many numbers to the appropiate number of digits
    """
    new_nums = []
    for i in nums:
        new_nums.append(round(i, digits))

    return new_nums


def sample_table(x, y, y_predict, columns=None, write_csv=False):
    """
    Parameters:
        x - pandas ds of X values with n attributes.
        y - pandas ds of Y values each 1 int (0, 1)
        y_predict - Y values from NN or SVM, confidence float from 0 - 1
        columns - List of attributes in X, not including Y or Y_predict
    """
    if not (len(x) == len(y) == len(y_predict)):
        raise ValueError("All datasets must be 1 dimensional and the same length.")
    # Only take the first 5 attributes
    if len(columns) > 5:
        columns = columns[:4]

    # Add the rest of the required columns
    columns.insert(0, "Sample #")
    columns.append("y_predict")
    columns.append("y")

    # Pretty print settings - debug
    # np.set_printoptions(precision=2, suppress=True)
    # pd.options.display.float_format = '{:.2f}'.format

    # Table settings
    table = PrettyTable()
    table.field_names = columns

    if write_csv:
        with open(write_csv, mode="w") as f:
            writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            # Write header
            writer.writerow(columns)
            for i, x_row in enumerate(x):
                # Round each number for pretty printing
                row = [
                    i,
                    *_clean_nums(*list(x_row[:4])),
                    *_clean_nums(y_predict[i]),
                    *_clean_nums(y[i]),
                ]
                # Write the rest of the data
                writer.writerow(row)
                table.add_row(row)
    else:
        for i, x_row in enumerate(x):
            # Round each number for pretty printing
            row = [
                i,
                *_clean_nums(*list(x_row[:4])),
                *_clean_nums(y_predict[i]),
                *_clean_nums(y[i]),
            ]
            table.add_row(row)

    print(table)
