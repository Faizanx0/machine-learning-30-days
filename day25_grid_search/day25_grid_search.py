"""
1. Build your Pipeline
Use numerical imputation and scaling, categorical imputation and one-hot encoding, and Logistic Regression.

2. Create a parameter grid
Test C values [0.01, 0.1, 1, 10, 100].

3. Create GridSearchCV
Use 5-fold StratifiedKFold and scoring='f1'.

4. Fit the search
Call grid_search.fit(X, y) for this practice exercise.

5. Print the results
Display best_params_, best_score_, and best_estimator_.

6. Explain the concepts
Explain what C controls and why we use classifier__C. - C is a Hyperparameter which tunes the model for best result, we use classifier__C becoz it have all the values of Hyperparameter for tuning model why it wrtten as classifier__C becoz the LogisticRegression is label as classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

data = {
    "Tenure": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15,
               18, 20, 22, 24, 30, 32, 36, 40, 42, 48],
    "MonthlyCharges": [90, 85, 80, 95, 70, 75, 65, 60, 55, 50,
                       60, 45, 50, 40, 55, 35, 45, 30, 40, 25],
    "SupportCalls": [5, 6, 4, 7, 3, 5, 2, 3, 2, 1,
                     4, 2, 3, 1, 2, 1, 2, 0, 1, 0],
    "ContractType": [
        "Monthly", "Monthly", "Monthly", "Monthly", "Yearly",
        "Monthly", "Yearly", "Yearly", "Yearly", "Yearly",
        "Monthly", "Yearly", "Yearly", "Yearly", "Monthly",
        "Yearly", "Monthly", "Yearly", "Monthly", "Yearly"
    ],
    "Churn": [1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df.drop("Churn", axis=1)
y = df["Churn"]

numerical_feature = ["Tenure","MonthlyCharges","SupportCalls"]
categorical_feature = ["ContractType"]

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
    
categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    
preprocessing = ColumnTransformer([
    ("num",numerical_pipeline,numerical_feature),
    ("cat",categorical_pipeline,categorical_feature)
    ])
    
model = Pipeline([
    ("preprocessing",preprocessing),
    ("classifier",LogisticRegression(max_iter=1000))
    ])
    
param_grid = {
    "classifier__C":[0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
    )

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    refit=True
    )
    
grid_search.fit(X,y)

print("Best Parameter:",grid_search.best_params_)
print("Best CV F1 Score:",grid_search.best_score_)
print("Best Estimator:\n",grid_search.best_estimator_)