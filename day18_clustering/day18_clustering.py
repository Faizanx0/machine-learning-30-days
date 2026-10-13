"""
1. Train and evaluate
Run the starter code. Report the accuracy and confusion matrix you actually get.

2. Predict a new student
Create a DataFrame for a student with StudyHours = 7, Attendance = 90, PreviousMarks = 75. Predict Pass/Fail using your trained model.

3. Understand probabilities
Use model.predict_proba(new_student) and explain what the returned numbers mean. - It Probability of both case Pass/Fail

4. Explain the concept
In your own words, explain why the algorithm is called Naive Bayes and what fit() does. - Becoz it use bayes thoerem and naive means feature which do not depend on other feature, fit() train the data on X_train,and y_train here it train on bayes formula

5. Experiment
Change the new student's StudyHours to 2 while keeping the other features unchanged. Predict again and compare. Don't assume the result before running the model. - In both case it give pass
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "StudyHours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 7, 3, 9, 5, 8, 3, 6, 9, 4
    ],
    "Attendance": [
        50, 55, 60, 65, 70, 75, 80, 90, 58, 68,
        85, 92, 62, 96, 78, 88, 64, 82, 98, 72
    ],
    "PreviousMarks": [
        35, 40, 45, 50, 55, 60, 65, 80, 42, 52,
        68, 75, 46, 88, 58, 78, 48, 67, 92, 54
    ],
    "Pass": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        1, 1, 0, 1, 1, 1, 0, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X=df[["StudyHours","Attendance","PreviousMarks"]]
y=df["Pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

model=GaussianNB()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Actual Value:",y_test.to_numpy())
print("Predicted Value:",prediction)

accuracy=accuracy_score(y_test,prediction)
print("Accuracy:",accuracy)
cm=confusion_matrix(y_test,prediction)
print("Confusion Matrix:\n",cm)

new_student = pd.DataFrame({
    "StudyHours": [7],
    "Attendance": [90],
    "PreviousMarks": [75]
})

new_prediction=model.predict(new_student)[0]
print("New Prediction:",new_prediction)
status = "Pass" if new_prediction==1 else "Fail"
print("Status:",status)
probabilities=model.predict_proba(new_student)
print("Probability:",np.round(probabilities,2))