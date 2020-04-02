# Cancer-Detection

## Method

Step 1: Data acquisition
A UCI Wisconsin dataset (1995) will be downloaded from the UCI machine learning
repository (<https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)>.
To train and evaluate a machine learning model, a sufficiently large dataset of mammogram
samples must be acquired. Within this dataset, features are computed from a digitized image of a
fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present
in the image. This dataset contains 569 samples, each with 32 attributes.

Step 2: Data visualization
Create a correlation map.
Determining relationships within the data can aid in deciding which machine learning
methods to use. Therefore, the relationships between their various attributes are visualized.

Step 3: Data pre-processing
Restructure the data and prepare for inputting into machine learning models.
This dataset is provided in a CSV file format. To accelerate the training process, all 569
samples are first loaded into working memory. The first attribute of each sample is an ID number
and is discarded. Then the remaining 31 features are split:
Y (attribute 31): This final attribute is the tumor classification, malignant and benign,
represented numerically as either 1 or 0. This feature is separated from the rest of the data. This is
referred to as the target feature.
X (attributes 1 30): These remaining features are the predictors (mean radius, mean
texture, mean perimeter, mean area, mean smoothness, etc.), which will be inputted into machine
learning models.
This pre-processing is a necessary to step to ensure data compatibility with industry
standard machine learning tools, such as TensorFlow and Scikit-learn. The data is then normalized,
slightly tweaking outliers to fit a more general distribution. Both X and Y are then split into two
smaller subsets. 80% of the data, 455 samples, is moved to a training set, and the remaining 20%,
114, samples are moved to a testing set. The training set will be directly inputted into both machine
learning models during their respective training phases. The remaining testing data will be used to
evaluate performance post training.

Step 4: Create an Artificial Neural Network.
Initialize the structure of an artificial neural network.
Modern machine learning frameworks greatly simplify model creation and evaluation. The
most recommended machine learning frameworks for artificial neural networks are Tensorflow
and Keras and are therefore preferred.
Generate a neural network with the following structure:
Layer 1 (Input) 30 nodes, sigmoidal activation function.
Layer 2 4 (Hidden) 512 nodes, rectified linear activation function.
Layer 5 (Output) 1 node, sigmoidal activation function.
Various other neural network sizes and activation functions are acceptable and can be
adjusted for optimal performance.

Step 5: Train the Artificial Neural Network.
Input training data and train the network until loss becomes acceptably low.
To learn, ANNs undergo several phases. The first phase, forward propagation, occurs when
the network is exposed to the training data. The data is passed through the entire network and a
prediction of whether a given sample is malignant or benign is generated. Initially, this prediction
is essentially random. Next, a loss function estimates how well the model performed, in this case
how accurately a sample was classified. Once the loss has been calculated, the information is
propagated backwards, hence the name: backpropagation. Starting from the output layers of the
network, the algorithm works backward, slightly adjusting edges and biases based on the loss. This
training cycle is repeated until the model accuracy is sufficiently high or the model loss is
sufficiently low.

Step 6: Create a Support Vector Machine
Initialize the structure of a support vector machine.
Modern machine learning frameworks greatly simplify model creation and evaluation. The
most recommended machine learning framework for support vector machines is Scikit-learn and
therefore preferred. Unlike artificial neural networks, there are no predisposed hyperparameters to
tune.

Step 7: Train the Support Vector Machine
Input training data into the support vector machine and train until loss becomes acceptably low.
Unlike most other classifiers, SVMs only focus on samples that are most difficult to
classify correctly. Other classifiers use all the data to evaluate relationships within the data. The
rationale behind a support vector machine is; if a model is sufficiently able to classify the most
challenging samples, then it will be able to classify all other samples with an even higher degree
of accuracy. The SVM model is explicitly searching for the best separating line between the two
classes of data. This is done by first searching for the two closest overlapping samples and finding
a line, typically linear, that connects them. The SVM then declares that the best separating line is
the line that bisects is perpendicular to the connecting line. This is repeated with many
overlapping samples until the number of samples misclassified is minimized or, more generally,
until the distance between the separating line and both classes of data is maximized.
Step 8: Evaluate both methods.
Use the testing dataset prepared in Step 3 to evaluate network performance.
During the data pre-processing step, the testing set was created to be intentionally isolated
from network training. Post training, this data into inputted into each machine learning method to
evaluate performance. A significantly lower performance in this testing data indicates a large
amount of overfitting has occurred. This secondary evaluation is essential to ensure a machine
learning models performance in a more generalized and real-world environment. These results
can then be used to generate heatmaps to easily describe, true positives, true negatives, false
positives, and false negatives.

## Results

![ANN_TEST](assets/ANN_TEST.png)
![ANN_TRAIN](assets/ANN_TRAIN.png)

![SVM_TEST](assets/SVM_TEST.png)
![SVM_TRAIN](assets/SVM_TRAIN.png)

<!-- <img src="<https://raw.githubusercontent.com/jarulsamy/Cancer-Detection/master/assets/ANN_TEST.png" alt="ANN_TEST" width=320 height=240>

<img src="<https://raw.githubusercontent.com/jarulsamy/Cancer-Detection/master/assets/ANN_TRAIN.png" alt="ANN_TRAIN" width=320 height=240>

<img src="<https://raw.githubusercontent.com/jarulsamy/Cancer-Detection/master/assets/SVM_TEST.png" alt="SVM_TEST" width=320 height=240>

<img src="<https://raw.githubusercontent.com/jarulsamy/Cancer-Detection/master/assets/SVM_TRAIN.png" alt="SVM_TRAIN" width=320 height=240> -->
