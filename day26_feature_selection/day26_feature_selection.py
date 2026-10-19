"""
Run the code and print the best parameters and best F1 score.
Change n_iter from 6 to 10. What changes? - Nothing change bith are same (None ,C=10,score=1.0)
Change scoring from 'f1' to 'accuracy'. Compare the selected settings. - Nothing changes both are same (None ,C=10,score=1.0)
Explain why RandomizedSearchCV can be faster than GridSearchCV.- The RandomizedSearchCV is mosre fast becoz in GridSearchCV it makes all possible combination like if we had 5 folds and two c so the total cases is 10 , idk how RandomizedSearchCV works and how it is diff from it but i know it is fast
Explain the meaning of classifier__C and n_iter=6. - classifier__C is used to get diff values of C to train over model, n_iter=6 means it take 6 iter to find most optimize one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)
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
    
preprocessor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_feature),
    ("cat",categorical_pipeline,categorical_feature)
    ])
    
model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression())
    ])

param_distributions = {
    "classifier__C":[0.001,0.01,0.1,1,10,100,1000],
    "classifier__class_weight":[None,"balanced"]
    }
    
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )

random_search = RandomizedSearchCV(
    estimator=model,
    n_iter=6,
    param_distributions=param_distributions,
    cv=cv,
    scoring="f1",
    random_state=42,
    refit=True
    )
    
random_search.fit(X,y)
print("Best Parameter:",random_search.best_params_)
print("Best F1 Score:",random_search.best_score_)

random_search_accuracy = RandomizedSearchCV(
    estimator=model,
    n_iter=6,
    param_distributions=param_distributions,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    refit=True
    )
random_search_accuracy.fit(X,y)
print("Best Parameter:",random_search_accuracy.best_params_)
print("Best Accuracy Score:",random_search_accuracy.best_score_)