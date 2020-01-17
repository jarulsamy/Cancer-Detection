from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from NeuralNet import step
from NeuralNet import train
from utils import load_cancer
from utils import pretty_cmatrix


# Only samples with 70% confidence are considered cancerous
THRESHOLD = 0.7

X_train, X_test, y_train, y_test = load_cancer()

# Timestamp for tensorboard logs
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path(logdir))

# Train the model
model = train(X_train, y_train, epochs=10)

# Predict and step each of the outputs for both datasets
# Stepping converts % confidence into a prediction based on thresh
y_predict = model.predict(X_test)
stepped_predict = []
for i in y_predict:
    stepped_predict.append(step(i, THRESHOLD))
pretty_cmatrix(stepped_predict, y_test, "ANN", "Test")


y_predict = model.predict(X_train)
stepped_predict = []
for i in y_predict:
    stepped_predict.append(step(i, THRESHOLD))
pretty_cmatrix(stepped_predict, y_train, "ANN", "TRAIN")

plt.show()
