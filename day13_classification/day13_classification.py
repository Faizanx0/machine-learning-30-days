"""
1. Explore the dataset
Print the shape, first five rows, and counts for Pass and Fail.
2. Prepare the data
Create X and y, then perform a stratified 75/25 train-test split.
3. Train and predict
Fit Logistic Regression and print actual versus predicted labels.
4. Evaluate
Calculate the confusion matrix, accuracy, precision, recall, and F1 score.
5. New student
Predict the class and probabilities for a student who studies 7 hours, has 90% attendance, and previous marks of 80.
6. Explain the results
In your own words, explain what the confusion matrix and F1 score tell you. - confusion matrix tell that the matrix no of actual and predicted value are true and false where f1 tell the value which come up 2*(precision*recall)/(precision+recall) 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7, 3, 9, 5, 8, 3, 6, 9, 4],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 85, 92, 62, 96, 78, 88, 64, 82, 98, 72],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 75, 46, 88, 58, 78, 48, 67, 92, 54],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Shape:",df.shape)
print("First five rows:\n",df.head())
print("Count Pass/Fail:\n",df["Pass"].value_counts())

X=df[["StudyHours","Attendance","PreviousMarks"]]
y=df["Pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
actual = y_test.to_numpy()
print("Actual value:",actual)
print("Predicted value",prediction)

cm = confusion_matrix(actual,prediction)
accuracy =  accuracy_score(actual,prediction)
precision = precision_score(actual,prediction)
recall = recall_score(actual,prediction)
f1 = f1_score(actual,prediction)
print("Confusion Matrix:\n",cm)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1 Score:",f1)

new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [85],
    "PreviousMarks": [70]
})

new_prediction = model.predict(new_student)
status = ["Pass" if new_prediction==1  else "Fail"]
new_proba = model.predict_proba(new_student)
print("Class predict:",new_prediction)
print("Student status:", status)
print("Predicion Probabilities:",np.round(new_proba,2))
