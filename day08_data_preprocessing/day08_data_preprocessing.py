
"""Exercise 1 — Understand the data

Using the hours and marks arrays above:
Print the shape of hours.
Print the shape of marks.
Explain in your own words why hours has an extra pair of brackets. - Because here hours is feature and which are multiple so skitlearn take in form of 2D array

Exercise 2 — Inspect the model

After training the model, print:
model.coef_
model.intercept_

What does the coefficient represent? - Here coefficient represent the slope of the equation of line 
What does the intercept represent?  - It represent the value of equation when x=0
Use both values to write the equation of your trained line.

Exercise 3 — Compare predictions

Print the actual and predicted marks next to each other.
Calculate MAE.
In your own words, explain what the MAE tells you. - MAE tell the avg error of prediction value it the actual val and predict val and div by 2 , hence lower the mae lower th error 

Exercise 4 — Make another prediction

Predict marks for a student who studies for 4.5 hours.
Use model.predict() and explain why you pass a 2D array.

Exercise 5 — Test your understanding

Answer without running code:
Is predicting marks regression or classification? - It is regression because here we were dealing with numerical data but in classification we classify the data whether it is or not  
Why don't we train the model on the testing data? - We use 80/20 rule in which 80% of data is for training and 20% is for testing, if we use this data for training we cant have data for test anymore
What is the difference between a feature and a target? - So the feature is value which is independent and its is the input to obtain the target where target is dependent on feature data.
Does a lower MAE always guarantee that a model will perform well on future students? Why or why not? - No lower MAE does not gurantee that the model will perform well for future student beacuse it give the avg value of actual and predicted val so for some cases it can judge there are several techinique like rms r2 etc"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

hours = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

marks = np.array([
    30, 35, 50, 55, 60,
    65, 70, 80, 85, 95
])

print("Hours shape:",hours.shape)
print("Marks shape:",marks.shape)

X_train,X_test,y_train,y_test = train_test_split(hours,marks,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)
print("Coefficient:",model.coef_)
print("Intercept:",model.intercept_)
print(f"Equation: Y = {model.coef_[0]:.2f} * X + {model.intercept_:.2f}")

predictions=model.predict(X_test)
print("Actual Value:",y_test)
print("Predicted Value:",np.round(predictions,2))

MAE = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error:",np.round(MAE,2))

new_hours=np.array([[4.5]])
predict_marks=model.predict(new_hours)
print("Predicted Marks:",np.round(predict_marks,2))