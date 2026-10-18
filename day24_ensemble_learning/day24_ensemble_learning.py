"""
1. Run five-fold cross-validation
Use the complete churn pipeline and StratifiedKFold from today's lesson.

2. Print all fold scores
Display the individual scores returned by cross_val_score().

3. Calculate the mean
Use scores.mean() to calculate average accuracy.

4. Calculate the standard deviation
Use scores.std() and explain what it tells you.

5. Try another scoring metric
Run cross_val_score() with scoring='f1' and compare its average with accuracy. - The f1 and avg both are same 1 

6. Answer the theory questions
Explain why cross-validation is useful and why accuracy alone can be misleading. - The cv is useful becoz it divide the data into folds and then train on each fold combination this gives the best and avg accuracy of diff folds, that why accuracy can be misleading  
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

X = df.drop("Churn",axis=1)
y = df["Churn"]

numerical_feature = ["Tenure","MonthlyCharges","SupportCalls"]
categorical_feature  = ["ContractType"]

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
    
categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    
preproccesor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_feature),
    ("cat",categorical_pipeline,categorical_feature)
    ])
    
model = Pipeline([
    ("preprocessing",preproccesor),
    ("classifier",LogisticRegression(max_iter=1000))
    ])
    
cross_val = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

scores = cross_val_score(model,X,y,cv=cross_val,scoring="accuracy")

print("Fold Scores:",scores)
print("Avg Accuracy:",scores.mean())
print("Standard Deviation:",scores.std())

f1_scores = cross_val_score(model,X,y,cv=cross_val,scoring="f1")

print("F1 Scores:",f1_scores)
print("Avg F1:",f1_scores.mean())
print("Standard Deviation:",f1_scores.std())