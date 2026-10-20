"""
Run the comparison code and record the accuracy, precision, recall and F1 for both models.
Compare the two confusion matrices. How many actual churn customers did each model detect? - Both models predict 1 actual churn customer
Explain why a model with 95% accuracy might still be poor at predicting churn. - Accuracy doesnot mean it can predicted the true values becoz if have 100 data out true 5 are not false and our model predict all true so the accuracy is 95% but it cant predict the false so instead of accuracy we use precision,recall and f1
Explain the difference between precision and recall. - Precision means Of all customers predicted to churn, how many actually churned? .Recall means Of all customers who actually churned, how many did the model detect?
Explain what class_weight='balanced' does. - By changing to balanced class weight the model is slightly bias to the less frequent class in our data for better results
Bonus: Change scoring in your Day 26 RandomizedSearchCV from accuracy to recall and compare the selected parameters. - After changing accuracy to recall the classifier__class_weight': 'balanced' is come instead of None
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

print("Class Distribution:",y.value_counts())

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
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

normal_model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
    ])
    
balanced_model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
        ))
    ])
    
normal_model.fit(X_train,y_train)
balanced_model.fit(X_train,y_train)

normal_predict = normal_model.predict(X_test)
balanced_predict = balanced_model.predict(X_test)

print("Actual Value:             ",y_test.to_numpy())
print("Normal Model Prediction:  ",normal_predict)
print("Balanced Model Prediction:",balanced_predict)
    
def comparison_model(model,actual,prediction):
    print("Model:",model)
    print("Accuracy:",np.round(accuracy_score(actual,prediction),2))
    print("Precision:",np.round(precision_score(actual,prediction,zero_division=0),2))
    print("Recall:",np.round(recall_score(actual,prediction,zero_division=0),2))
    print("F1:",np.round(f1_score(actual,prediction,zero_division=0),2))
    print("Confusion Matrix:\n",confusion_matrix(actual,prediction))
    print("-"*50)

comparison_model("normal_model",y_test,normal_predict)
comparison_model("balanced_model",y_test,balanced_predict)