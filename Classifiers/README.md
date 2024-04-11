# Classifiers Overview

This section provides an overview of the classifiers utilized in the project, along with their respective Python scripts.

## Decision Tree Classifier

**Script:** [decisiontree_classifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/DecisionTree/decisiontree_classifier.py)

The Decision Tree Classifier is a non-parametric supervised learning method used for classification tasks. It works by recursively partitioning the input space into regions based on feature values, with the aim of minimizing impurity or maximizing information gain at each split. The script implements the Decision Tree Classifier algorithm, providing functionality for training and making predictions based on input features.

## Gradient Boosting Classifier

**Script:** [gradientboosting.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/GradienBoosting/gradientboosting.py)

The Gradient Boosting Classifier is an ensemble learning method that combines the predictions of multiple weak learners, typically decision trees, to improve overall predictive accuracy. It builds a series of decision trees sequentially, with each tree correcting the errors of the previous ones. The script implements the Gradient Boosting Classifier algorithm, offering functionalities for training and prediction.

## K-Nearest Neighbors (KNN) Classifier

**Script:** [kneighborsclassifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/KNeighborsClassifier/kneighborsclassifier.py)

The K-Nearest Neighbors (KNN) Classifier is a simple and intuitive machine learning algorithm used for classification tasks. It classifies a data point based on the majority class among its k nearest neighbors in the feature space. The script provides an implementation of the KNN algorithm, allowing for training and prediction based on the nearest neighbors of input samples.

## Linear Support Vector Classifier (LinearSVC)

**Script:** [linearsvc.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/LinearSVC/linearsvc.py)

The Linear Support Vector Classifier (LinearSVC) is a linear classifier that separates classes by defining a hyperplane in the feature space. It aims to maximize the margin between different classes while minimizing classification errors. The script implements the LinearSVC algorithm, offering functionalities for training and prediction.

## Random Forest Classifier

**Script:** [randomforestclassifier.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Classifiers/Random_Forest/randomforestclassifier.py)

The Random Forest Classifier is an ensemble learning method that builds a collection of decision trees during training and combines their predictions through voting or averaging to improve accuracy and robustness. It is particularly effective for handling high-dimensional data and complex classification tasks. The script implements the Random Forest Classifier algorithm, providing functionalities for training and prediction.

# Folders

## Trained Models

[Trained_Models](https://github.com/sankalp-s/MSC_Thesis/tree/main/Classifiers/Trained_Models)

This folder contains the trained models generated using scripts for various classifiers. The models are trained to classify actions in the Super Mario World game environment based on the extracted features.
You can load these trained models into your Python environment using TensorFlow or scikit-learn library to perform classification tasks on new data.



