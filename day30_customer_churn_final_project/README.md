# Customer Churn Prediction

A beginner-friendly Machine Learning project that predicts whether a customer may leave a service using Logistic Regression.

## Project Objectives

- Build a complete classification workflow.
- Compare a baseline model with a hyperparameter-tuned model.
- Evaluate predictions using classification metrics.
- Save and reuse the trained model.

## Features Used

| Feature | Description |
|---|---|
| Tenure | Length of time the customer has used the service |
| MonthlyCharges | Customer's monthly service charge |
| SupportCalls | Number of support calls |
| ContractType | Monthly or yearly contract |
| Churn | Target: 1 = churn, 0 = stay |

## Tools and Libraries

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib

## Workflow

1. Explore the data.
2. Split the data into training and test sets.
3. Preprocess numerical and categorical features.
4. Train a baseline Logistic Regression model.
5. Tune hyperparameters using RandomizedSearchCV and three-fold stratified cross-validation.
6. Compare performance using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Save the tuned pipeline and use it for new predictions.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python train_model.py
```

Make a prediction:

```bash
python predict.py
```

## Limitations

The current dataset contains only 20 artificial customer records, including just 6 churn cases. Results can vary significantly with different splits and model settings. These scores do not establish real-world predictive performance.

## Future Improvements

- Use a larger, realistic customer churn dataset.
- Evaluate on a separate, representative test set.
- Compare additional classification models.
- Explore class imbalance and decision thresholds.
- Add a simple web interface for predictions.

## Project Status

Completed as a beginner Machine Learning learning project. Further testing is required before any real-world use.
