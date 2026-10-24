"""
1.Build the dataset
Use the customer dataset from Step 1.

2. Separate features and target
Create X and y, with Churn as the target.

3. Split the data
Use 25% for testing, random_state=42, and stratify=y.

4. Build numerical preprocessing
Use median imputation followed by StandardScaler.

5. Build categorical preprocessing
Use most-frequent imputation followed by OneHotEncoder.

6. Build and evaluate the Pipeline
Combine the ColumnTransformer and LogisticRegression, then print accuracy and the confusion matrix.

7. Predict a new customer
Predict churn for Tenure=2, MonthlyCharges=95, SupportCalls=6, ContractType='Monthly'. Pass a DataFrame with the same feature columns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


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

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25,stratify=y)

numerical_features = ["Tenure","MonthlyCharges","SupportCalls"]
categorical_features = ["ContractType"]

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])
    
categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])
    
preprocessor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_features),
    ("cat",categorical_pipeline,categorical_features)
    ])
    
model = Pipeline([
    ("preprocessing",preprocessor),
    ("classifier",LogisticRegression(max_iter=1000))
    ])

model.fit(X_train,y_train)

prediction = model.predict(X_test)
print("Predicted Value:",prediction)
print("Actual Value:",y_test.to_numpy())
accuracy = accuracy_score(y_test,prediction)
print("Accuracy:",accuracy)
cm = confusion_matrix(y_test,prediction)
print("Confusion Matrix:\n",cm)

New_Data = pd.DataFrame({
    "Tenure": [2],
    "MonthlyCharges" : [95],
    "SupportCalls" : [6],
    "ContractType" : "Monthly"
    })
    
new_prediction = model.predict(New_Data)[0]
print("Customer Prediction:",new_prediction)
if new_prediction==1:
    print("Customer will Leaves!")
else:
    print("Customer will Stays!")