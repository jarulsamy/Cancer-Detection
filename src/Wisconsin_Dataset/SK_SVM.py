import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils import load_cancer
from utils import pretty_cmatrix

X_train, X_test, y_train, y_test = load_cancer()
# Train
svc_model = SVC(verbose=True)  # C-Support Vector Classification
# svc_model = LinearSVC(verbose=True)  # Linear Support Vector Classification
svc_model.fit(X_train, y_train)

# Confusion Matrix for both Test and training sets
y_predict = svc_model.predict(X_test)
pretty_cmatrix(y_predict, y_test, "SVM", "Test")
y_predict = svc_model.predict(X_train)
pretty_cmatrix(y_predict, y_train, "SVM", "Train")

plt.show()
