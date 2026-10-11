"""
1. Build the model
Write the imports, create the DataFrame, separate X and y, and split the data.

2. Train the tree
Create a DecisionTreeClassifier with max_depth=3 and random_state=42, then call fit().

3. Evaluate it
Print the test predictions, accuracy, and confusion matrix.

4. Predict a new student
Use StudyHours=7, Attendance=, PreviousMarks=75 and print the predicted class.

5. Experiment
Change max_depth to 1, then 5. Compare the accuracies. Explain what you observe."""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7, 3, 9, 5, 8, 3, 6, 9, 4],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 85, 92, 62, 96, 78, 88, 64, 82, 98, 72],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 75, 46, 88, 58, 78, 48, 67, 92, 54],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X=df[["StudyHours","Attendance","PreviousMarks"]]
y=df["Pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model = DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(X_train,y_train)

predictions = model.predict(X_test)
print("Actual Value:",y_test.to_numpy())
print("Predicted Value",predictions)
accuracy = accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix:\n",cm)

new_student = pd.DataFrame({
    "StudyHours": [7],
    "Attendance": [90],
    "PreviousMarks": [75]
})

new_prediction = model.predict(new_student)
status = "Pass" if new_prediction else "Fail"
print("New Prediction:",new_prediction)
print("Status:",status)
