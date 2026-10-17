"""1. Run the model
Submit the accuracy, confusion matrix, precision, recall and F1 score produced by your code.

2. Predict a new customer
Create a DataFrame for Tenure = 2, MonthlyCharges = 95, SupportCalls = 6. Predict whether the customer will churn.

3. Check probabilities
Use model.predict_proba(new_customer) and explain which probability corresponds to churn. - For churn0: 0 and churn1: 1

4. Interpret the model
Explain why accuracy alone may not be enough when a company wants to identify customers who may leave. - Accuracy cant define things becoz in real life there are many scenerio that changes the result

5. Explain the workflow
In your own words, explain X, y, fit(), predict() and stratify=y. - X is input or feature , y is the output or target , fit() train the model on both X&y , predict() predict the corresponding output and stratify=y helps the equallyy distribution of output both cases in test cases
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

data = {
    "Tenure": [
        1, 2, 3, 4, 5, 6, 8, 10, 12, 15,
        18, 20, 22, 24, 30, 32, 36, 40, 42, 48,
        3, 7, 9, 14, 17, 25, 28, 35, 38, 45
    ],
    "MonthlyCharges": [
        90, 85, 80, 95, 70, 75, 65, 60, 55, 50,
        60, 45, 50, 40, 55, 35, 45, 30, 40, 25,
        100, 95, 85, 80, 75, 70, 65, 60, 50, 35
    ],
    "SupportCalls": [
        5, 6, 4, 7, 3, 5, 2, 3, 2, 1,
        4, 2, 3, 1, 2, 1, 2, 0, 1, 0,
        7, 5, 6, 4, 5, 3, 4, 2, 3, 1
    ],
    "Churn": [
        1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

X=df[["Tenure","MonthlyCharges","SupportCalls"]]
y=df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.25,stratify=y)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print("Predicted Value:",prediction)
print("Actual Value:",y_test.to_numpy())

cm = confusion_matrix(y_test,prediction)
accuracy = accuracy_score(y_test,prediction)
precision = precision_score(y_test,prediction)
recall = recall_score(y_test,prediction)
f1 = f1_score(y_test,prediction)
print("Confusion Matrix:\n",cm)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)

New_customer = pd.DataFrame({
    "Tenure" : [2],
    "MonthlyCharges" : [95],
    "SupportCalls" : [6]
})

new_prediction = model.predict(New_customer)[0]
print("Customer Prediction:",new_prediction)
if new_prediction == 1:
    print("Predicted outcome: Customer may churn")
else:
    print("Predicted outcome: Customer may stay")
print("Customer Probability:",np.round(model.predict_proba(New_customer),2))