"""
Run the complete code and confirm that churn_model.joblib is created.
Load the saved model and predict for a second new customer.
Print predictions from both the original model and the loaded model for the same test data. Check whether they match.
Explain the difference between joblib.dump() and joblib.load(). - The dump() is used to save model in our sytem and the load() is used to load  the model from our system
Explain why saving the entire pipeline is better than saving only Logistic Regression. - Becoz the pipeline helps for all model type preprocessing like scaling,encoding etc, so if we save only pipeline we also change the model type but if we save LogisticRegression we rewrite the whole pipeline for this model and other model types too
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    ("classifier",LogisticRegression(max_iter=1000,class_weight="balanced"))
    ])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

model.fit(X_train,y_train)

print("----Model Trained Successfully----")

prediction = model.predict(X_test)
print("Predicted Value:",prediction)
print("Accuracy:",accuracy_score(y_test,prediction))

joblib.dump(model,"churn_model.joblib")
print("----Model Saved Successfully----")

load_model = joblib.load("churn_model.joblib")
print("----Model Load Successfully----")


new_customer = pd.DataFrame([{
    "Tenure": 7,
    "MonthlyCharges": 80,
    "SupportCalls": 4,
    "ContractType": "Monthly"
}])

new_prediction = load_model.predict(new_customer)[0]
print("New Customer Prediction:",new_prediction)

if new_prediction==1:
    print("Customer will Leave!\n")
else:
    print("Customer will Stay!\n")
    
print("Original Model Prediction:",model.predict(X_test))
print("Loaded Model Prediction:  ",load_model.predict(X_test))
print("Actual Value:             ",y_test.to_numpy())