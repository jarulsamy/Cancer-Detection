# Cancer-Detection

Machine learning aids in many of our day-to-day ordeals in normal life. Some of the most powerful uses of machine learning are in the medical field. In this repository, the viability of artificial neural networks and support vector machines in tumor malignancy classification is tested.

## Dependencies

-   Python **3.6+**

-   All other dependencies should be documented in [requirements.txt](src/requirements.txt).

## Simplified Pipeline

Step 1: Data acquisition

-   A publically available UCI Wisconsin breast cancer dataset is downloaded. Within this dataset, features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Step 2: Data visualization

-   To effectively identifiy relationships within the data itself, a heatmap of all 31 features within the data is created.

Step 3: Data pre-processing

-   Attributes are seperated according to the following specifications:

    -   The first attribute of each sample is an ID number and is discarded

    -   Y (attribute 31): This final attribute is the tumor classification, malignant and benign, represented numerically as either 1 or 0. This feature is separated from the rest of the data. This is referred to as the target feature.

    -   X (attributes 1 - 30): These remaining features are the predictors (mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.), which will be inputted into machine learning models

-   The data is also split into to sets, training and testing.

Step 4: Create and train an Artificial Neural Network.

-   Modern neural networks come in many shapes and sizes. For this project, a simple 5 layer neural network was chosen for it's decent performance with minimal computational overhead.
    > More details about specific network architecture is in the [official writeups](docs/README.md).

Step 6: Create and train a Support Vector Machine

-   The SVM model is explicitly searching for the best separating line between the two classes of data. This is done by first searching for the two closest overlapping samples and finding a line, typically linear, that connects them. The SVM then declares that the best separating line is the line that bisects  is perpendicular to  the connecting line.  This is repeated with many overlapping samples until the number of samples misclassified is minimized or, more generally, until  the  distance  between  the  separating  line  and  both  classes  of  data  is  maximized.

Step 7: Evaluate and visualize

-   Post Training, data from both data sets is stepped through both machine learning methods. This evaluation data is used generate the result plots.

> A more detailed formal publication is written [here](docs/README.pdf).

## Results

<img src="assets/ANN_TEST.png">
<img src="assets/ANN_TRAIN.png">
<img src="assets/SVM_TEST.png">
<img src="assets/SVM_TRAIN.png">

## Running

Reproducibility is a priority! Everything related to this research is freely published here. So go ahead, run some classifications!

The `MRI` directory is a failed attempt at classifying MRI images and is not currently working.

The `Wisconsin_Dataset` directory contains all code related to this publication.

1.  Install dependencies

        cd src/
        pip install -r requirements.txxt

2.  Run the neural net

        cd src/Wisconsin_Dataset
        python SK_ANN.py

3.  Run the SVM

        cd src/Wisconsin_Dataset
        python SK_SVM.py

> In both step 2 & 3, the UCI dataset should be automatically be loaded with `scikit-learn`.

## Hyperparameters

### ANN

The specific neural network size and architecture can be directly altered from `src/Wisconsin_Dataset/NeuralNet.py`. The `prep_model()` as you see necessary.
