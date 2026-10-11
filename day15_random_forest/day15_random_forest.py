"""
1. Create and train Random Forest
Use the student dataset and train a RandomForestClassifier with 100 trees and max_depth=3.

2. Evaluate the model
Print actual values, predictions, accuracy, and confusion matrix.

3. Predict a new student
Predict the class for StudyHours=7, Attendance=90, PreviousMarks=75.

4. Compare both algorithms
Train a Decision Tree using the same split and compare its accuracy with Random Forest.

5. Answer in your own words
Why might combining 100 trees be more useful than using just one tree? Is Random Forest always more accurate? - Beacause one tree can be incorrect so we check for multiple tree and make forest to ans into a single one that make random forset more accurate becoz it have group of trees predictions
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
forest_model = RandomForestClassifier(n_estimators=100,max_depth=3,random_state=42)
forest_model.fit(X_train,y_train)

forest_predictions=forest_model.predict(X_test)
print("Actual Value:",y_test.to_numpy())
print("Predicted Value:",forest_predictions)
forest_accuracy=accuracy_score(y_test,forest_predictions)
print("Accuracy:",forest_accuracy)
forest_cm=confusion_matrix(y_test,forest_predictions)
print("Confusion Matrix:\n",forest_cm)

new_student = pd.DataFrame({
    "StudyHours": [7],
    "Attendance": [90],
    "PreviousMarks": [75]
})

forest_prediction = forest_model.predict(new_student)
print("Forest Prediction:",forest_prediction)
print("Status:","Pass" if forest_prediction else "Fail")

tree_model = DecisionTreeClassifier(max_depth=3,random_state=42)
tree_model.fit(X_train,y_train)
tree_predictions = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test,tree_predictions)

tree_prediction = tree_model.predict(new_student)
print("Tree Prediction:",tree_prediction)
print("Status:","Pass" if tree_prediction else "Fail")

print("Forest Accuracy:",forest_accuracy)
print("Tree Accuracy:",tree_accuracy)