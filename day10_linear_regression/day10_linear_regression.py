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
