"""1. Build the SVM model
Create the dataset, split it, and train SVC(kernel='linear') using StandardScaler and make_pipeline.

2. Evaluate the model
Print actual values, predictions, accuracy, and confusion matrix.

3. Predict a new student
Use StudyHours=7, Attendance=90, PreviousMarks=75 and print Pass or Fail. 

4. Experiment with kernels
Change kernel='linear' to kernel='rbf'. Compare test accuracy. - Nothing change the accuracy remains 0.75

5. Answer the concepts
What are support vectors? Why does SVM benefit from feature scaling? - Support vector give the direction in which the hyperline will be drawn, SVM benefit from scaling becoz the hyperline is drawn btw class which is meant by numercial formula
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7, 3, 9, 5, 8, 3, 6, 9, 4],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 68, 85, 92, 62, 96, 78, 88, 64, 82, 98, 72],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 75, 46, 88, 58, 78, 48, 67, 92, 54],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[["StudyHours", "Attendance", "PreviousMarks"]]
y = df["Pass"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
    )
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Actual Value:",y_test.to_numpy())
print("Predicted Value:",predictions)
accuracy=accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)
cm = confusion_matrix(y_test,predictions)
print("Confusion Matrix:\n",cm)

new_student = pd.DataFrame({
    "StudyHours": [7],
    "Attendance": [90],
    "PreviousMarks": [75]
})

new_prediction = model.predict(new_student)[0]
print("New Prediction:",new_prediction)
status = "Pass" if new_prediction==1 else "Fail"
print("Status:",status)