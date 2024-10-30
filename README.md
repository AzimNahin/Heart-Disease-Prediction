# Heart Disease Prediction

This repository contains code and resources for predicting heart disease based on patient data. Using machine learning and deep learning models, this project aims to identify the likelihood of heart disease presence based on various health metrics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Models Used](#models-used)
- [Installation](#installation)
- [Results](#results)
- [Contributor](#contributor)

## Overview

Heart disease is a leading cause of death globally, and early detection can greatly improve patient outcomes. This project applies various machine learning and deep learning techniques to classify patients as at-risk for heart disease based on medical and lifestyle data.

## Features

- **Data Preprocessing**: Handles missing values, normalizes features, and applies encoding as required.
- **Machine Learning Models**: Includes models like Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and a Dense Neural Network (DNN).
- **Evaluation Metrics**: Model performance is evaluated using metrics such as accuracy, precision, recall, F1 score, and AUC.
- **Reproducible Results**: Code includes random seed settings to ensure consistent results.

## Models Used

- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**
- **Dense Neural Network (DNN)**

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AzimNahin/Heart-Disease-Prediction.git
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

## Results

The models in this project are evaluated using several key metrics:

- **Accuracy**: Measures the overall correctness of the model in identifying cases with and without heart disease.
- **Precision**: Indicates the proportion of positive predictions that were correct.
- **Recall**: Measures the model's effectiveness in identifying true positive cases.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced metric for imbalanced datasets.
- **AUC (Area Under the Curve)**: Reflects the model's performance across different classification thresholds, measuring the trade-off between true positive and false positive rates.

Results of each model can be found in the [results](results/) folder. It includes a comparison of various models used, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and Dense Neural Network (DNN), showing their strengths and weaknesses in heart disease prediction.

## Contributors  
- [AzimNahin](https://github.com/AzimNahin)

