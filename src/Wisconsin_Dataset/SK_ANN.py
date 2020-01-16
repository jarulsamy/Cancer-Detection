import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from NeuralNet import binary_step
from NeuralNet import prep_model
from NeuralNet import train
from utils import load_cancer
from utils import pretty_cmatrix

# Suppress tensorflow verbose output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

X_train, X_test, y_train, y_test = load_cancer()

# Timestamp for tensorboard logs
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path(logdir))

# Train the model
model = train(X_train, y_train, epochs=3)

# Predict and step each of the outputs for both datasets
y_predict = model.predict(X_test)
stepped_predict = []
for i in y_predict:
    stepped_predict.append(binary_step(i))
pretty_cmatrix(stepped_predict, y_test, "ANN", "Test")

y_predict = model.predict(X_train)
stepped_predict = []
for i in y_predict:
    stepped_predict.append(binary_step(i))
pretty_cmatrix(stepped_predict, y_train, "ANN", "TRAIN")

plt.show()
