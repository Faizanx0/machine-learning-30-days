"""Exercise 1 — Calculate errors manually

Given:

actual = np.array([50, 60, 70, 80])
predicted = np.array([48, 63, 68, 75])

Calculate manually using NumPy:

Absolute errors
MAE
Squared errors
MSE
RMSE

Don't use scikit-learn for this exercise.

Exercise 2 — Use scikit-learn

Using the student dataset:

Create X and y.
Perform the train/test split.
Train LinearRegression.
Generate predictions.
Calculate:
MAE
MSE
RMSE
R²

Print all four.

Exercise 3 — Explain your model

After getting your results, answer:

What does your MAE mean in terms of marks? - MAE give mean of the error marks of the prediction 
Why is RMSE different from MAE? - In RMSE we square root the mse where in mae we take mean of the absolute diff
Why is MSE usually larger numerically than RMSE? - Yes because in RMSE we take the val of mse and sqrt to this so the val of rmse decresases
What does your R² tell you? - It tell the how the value is differ from base line if 1 so the model is perfect if 0 then some error if in negatie so model is not accurate
Does your R² mean your model is "X% accurate"? Explain. - If r2 is 0.2 doesnot means 20% accrate beacuse it show the diflection not the real prediction 

Exercise 4 — Compare two predictions

Use:

actual = np.array([50, 60, 70, 80])

model_A = np.array([51, 59, 71, 79])

model_B = np.array([40, 65, 75, 90])

Calculate MAE and RMSE for both models.

Then answer:

Which model would you choose based on these metrics, and why? - I would choose model_A because its MAE and rmse is perfect as compared model_B

Exercise 5 — Think like an ML engineer

Suppose you get:

Training MAE = 1.2
Testing MAE  = 8.7

What could be happening? - Here our model is perfect for Training data but not for Testing data

Explain in your own words."""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


actual = np.array([50, 60, 70, 80])
predicted = np.array([48, 63, 68, 75])

diff=actual-predicted
absolute=np.abs(diff)
mae=np.mean(absolute)
square=diff**2
mse=np.mean(square)
rmse=np.sqrt(mse)
print("Absolute errors:",absolute)
print("MAE:",mae)
print("Squared errors:",square)
print("MSE:",mse)
print("RMSE:",np.round(rmse,2))

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 7, 3, 9, 5],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95, 62, 78, 88, 92, 72, 96, 82],
    "PreviousMarks": [35, 40, 45, 50, 55, 60, 65, 80, 42, 52, 68, 72, 48, 88, 58],
    "FinalMarks": [38, 42, 48, 53, 59, 65, 72, 84, 43, 56, 70, 75, 50, 91, 62]
}

df = pd.DataFrame(data)

X=df[["StudyHours","Attendance","PreviousMarks"]]
y=df["FinalMarks"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print("Predictions:",np.round(predictions,2))
print("Actual value:",y_test.to_numpy())

mae=mean_absolute_error(y_test,predictions)
print("MAE:",np.round(mae,2))
mse=mean_squared_error(y_test,predictions)
print("MSE:",np.round(mse,2))
rmse=np.sqrt(mse)
print("RMSE",np.round(rmse,2))
r2=r2_score(y_test,predictions)
print("R2:",np.round(r2,2))

actual = np.array([50, 60, 70, 80])
model_A = np.array([51, 59, 71, 79])
model_B = np.array([40, 65, 75, 90])

def Eval_model(actual,model,name):
    diff=np.abs(actual-model)
    mae=np.mean(diff)
    rmse=np.sqrt(np.mean(diff**2))
    print(f"{name} MAE:",np.round(mae,2))
    print(f"{name} RMSE:",np.round(rmse,2))

Eval_model(actual,model_A,"model_A")
Eval_model(actual,model_B,"model_B")