"""
Exercise 1 — Understand the problem

For each one, write Regression or Classification:

1. Predict a student's final marks - Regression
2. Predict whether an email is spam - Classification
3. Predict house price - Regression
4. Predict whether a customer will leave - Classification
5. Predict tomorrow's temperature - Regression
6. Predict whether a student passes - Classification

Exercise 2 — Build the model

Use this dataset:

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 85, 92],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 75],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
}

Do these yourself:
    Create DataFrame
    Create X
    Create y
    Split into train/test
    Create LogisticRegression
    Train the model
    Predict
    Print predictions and actual values
    Calculate accuracy

Exercise 3 — Probability

Add:
model.predict_proba(X_test)
Look carefully at the output.
Explain what this means:
[0.25, 0.75] - It explain 25 percentage of fail and 75% of pass

Exercise 4 — New Student

Create a new student:

StudyHours = 6
Attendance = 85
PreviousMarks = 70

Use the model to predict whether the student passes.

Hint:

new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [85],
    "PreviousMarks": [70]
})

Then use:
model.predict(new_student)
and:
model.predict_proba(new_student)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 85, 92],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 75],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["StudyHours","Attendance","PreviousMarks"]]
y = df["Pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
prediction = model.predict(X_test)

print("Actual Value:",y_test.to_numpy())
print("Predicted Value:",prediction)

accuracy = model.predict_proba(X_test)
print("Prediction Accuracy:\n",np.round(accuracy,2))

new_student = pd.DataFrame({
    "StudyHours" : [6],
    "Attendance" : [85],
    "PreviousMarks" : [70]
})

new_prediction = model.predict(new_student)
new_accuracy = model.predict_proba(new_student)
print("New Prediction:",new_prediction,)
status=["Pass" if new_prediction==1 else "Fail"]
print("Student Status:",status)
print("New Accuracy:",np.round(new_accuracy,2))