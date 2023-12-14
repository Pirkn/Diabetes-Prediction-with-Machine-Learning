# Diabetes Prediction with Machine Learning

This repository contains a machine learning model for predicting diabetes based on various health features. The model is implemented in Python using popular libraries such as scikit-learn and pandas.

## Contents

- [Dataset](#dataset)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project is [diabetes.csv](diabetes.csv), which contains various health metrics and a binary target variable indicating the presence or absence of diabetes.

## Usage

To run the code and reproduce the results, follow these steps:

1. Clone the repository to your local machine.
2. Make sure you have the required libraries installed (listed in the code).
3. Run the Jupyter Notebook [DiabetesPred.ipynb](DiabetesPred.ipynb).

## Machine Learning Models

We experimented with several machine learning models, including:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)

Each model's performance is evaluated, and hyperparameter tuning is performed using GridSearchCV.

## Results

Here are the accuracy results for each model on the test dataset:

- K-Nearest Neighbors (KNN): [86.21]%.
- Logistic Regression: [82.76]%.
- Decision Tree: [82.76]%.
- Random Forest: [86.21]%.
- Naive Bayes: [83.62]%.
- Support Vector Machine (SVM): [85.34]%.

## Contributing

Contributions to this project are welcome! Feel free to open issues or pull requests.
