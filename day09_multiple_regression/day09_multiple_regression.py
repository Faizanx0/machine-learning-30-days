"""
Exercise 1 — Understand the dataset

Using df:
Display the first five rows.
Print the shape of X and y.
Explain why X has three columns but y has only one dimension.

Exercise 2 — Understand the model

After training:
Print the intercept.
Print all three coefficients with their feature names.
Write the equation of your trained model using the actual values you obtain.

Exercise 3 — Evaluate predictions

Print actual test marks beside predicted marks.
Calculate MAE.
Explain what your MAE means in terms of marks.

Exercise 4 — Predict a new student

Use the trained model to predict final marks for a student with:

Study hours 6
Attendance 90%
Previous marks 75
Create a DataFrame with the correct column names and pass it to model.predict().

Exercise 5 — Concept questions

Answer in your own words:
What is the difference between Simple and Multiple Linear Regression?
What does a coefficient represent when the other features are held constant? 
Why do we keep test data separate from training data?
If the training MAE is low but the test MAE is high, what might that indicate? 
Does a positive coefficient prove that a feature causes marks to increase? Explain. 
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7, 3, 9, 5],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95, 62, 78, 88, 92, 72, 96, 82],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 72, 48, 88, 58],
    "FinalMarks": [38, 42, 48, 53, 59, 65, 72, 84, 43, 56, 70, 75, 50, 91, 62]
}

df = pd.DataFrame(data)
X=df[["StudyHours","Attendance","PreviousMarks"]]
Y=df["FinalMarks"]
print("First 5 rows:\n",df.head())
print("X Shape:",X.shape)
print("Y Shape:",Y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)

print("Intercept:",np.round(model.intercept_,2))
for feature, coef in zip(X.columns, model.coef_):
    print(feature, "coefficient:", np.round(coef,2))
print(f"Equation: Y = {X.columns[0]} * {np.round(model.coef_[0],2)} + {X.columns[1]} * {np.round(model.coef_[1],2)} + {X.columns[2]} * {np.round(model.coef_[2],2)} + {np.round(model.intercept_,2)}")    

prediction = model.predict(X_test)
print("Actual Value:",y_test.to_numpy())
print("Predicted Value:",np.round(prediction,2))
MAE = mean_absolute_error(y_test,prediction)
print("Mean Absolute Error:",np.round(MAE,2))

new_student =pd.DataFrame({
    "StudyHours":[6],
    "Attendance": [90],
    "PreviousMarks":[75]
})
print("Predicted Marks:",np.round(model.predict(new_student),2))