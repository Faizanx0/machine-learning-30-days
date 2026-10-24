"""
1. Create the DataFrame
Use the four-row customer dataset shown above.

2. Check missing values
Use isnull().sum() to find missing values.

3. Fill missing charges
Fill missing MonthlyCharges using the median.

4. Encode ContractType
Use pd.get_dummies() with dtype=int.

5. Create a new feature
Create HighCharges: 1 if MonthlyCharges > 70, otherwise 0. Do this before scaling.

6. Explain the concepts
In your own words: why encode text, and why should scaling be fitted on training data only? - encoding text is most imp because there some missing data, some error, diff value like RAMESH & ramesh, Scaling on training data only is uselsss becoz after u want fit the values of train and test are very diff so u scale on test data also so it are fitted in model 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {
    "Tenure": [2, 24, 12, 36],
    "MonthlyCharges": [95, 40, None, 30],
    "ContractType": ["Monthly", "Yearly", "Monthly", "Yearly"],
    "SupportCalls": [6, 1, 3, 0],
    "Churn": [1, 0, 0, 0]
}

df = pd.DataFrame(data)
print("Missing Value:\n",df.isnull().sum())
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
df = pd.get_dummies(df,columns=["ContractType"],dtype=int)
df["HighCharges"] = (df["MonthlyCharges"] > 70).astype(int)

X = df[["Tenure","MonthlyCharges","SupportCalls","ContractType_Monthly","ContractType_Yearly","HighCharges"]]
y = df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)