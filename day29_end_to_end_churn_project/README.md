# Customer Churn Prediction

## Project Overview

This project uses Machine Learning to predict whether a customer is likely to leave a service (churn) or stay.

It compares a baseline Logistic Regression model with a tuned Logistic Regression model to understand how hyperparameter tuning affects model performance.

## Features

- Data exploration and class distribution analysis
- Numerical and categorical data preprocessing
- Baseline Logistic Regression model
- Hyperparameter tuning using RandomizedSearchCV
- Three-fold stratified cross-validation
- Model evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
- Saving and loading the trained model using Joblib
- Predictions and estimated churn probabilities for new customers

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Machine Learning Workflow

1. Prepare the dataset and separate features from the target.
2. Split the data into training and test sets.
3. Build preprocessing and classification pipelines.
4. Train a baseline model.
5. Tune hyperparameters using cross-validation.
6. Compare model performance on the held-out test set.
7. Save and reload the selected model.
8. Predict churn for new customers.

## Limitations

- The project currently uses a small, artificial dataset containing only 20 customer records.
- The evaluation scores are not reliable estimates of real-world performance.
- Predictions depend on the data patterns learned during training.
- The model has not been validated on a large, real-world customer dataset.

## Future Improvements

- Use a larger, publicly available customer churn dataset.
- Evaluate the model on unseen real-world-like data.
- Improve feature selection and model comparison.
- Explore ways to handle class imbalance.

## Disclaimer

This is a beginner educational project. Its predictions should not be used to make real business decisions without further testing and validation.