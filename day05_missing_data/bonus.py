"""
Create a final cleaned DataFrame that:

Removes exact duplicate rows.
Standardizes department names.
Fills missing ages with the median.
Fills missing marks with the mean.
Prints the final dataset and confirms that no missing values remain."""


import pandas as pd
import numpy as np

data = {
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir", "Sara"],
    "Age": [19, 20, np.nan, 21, 20, 20],
    "Marks": [85, np.nan, 67, 78, 95, np.nan],
    "Department": ["CSE", "IT", "cse", "it", np.nan, "IT"]
}

df = pd.DataFrame(data)
print("Missing values:",df.isnull().sum().sum())
df=df.drop_duplicates()
df["Department"]=df["Department"].str.strip().str.upper().fillna("CSE")

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Marks"]=df["Marks"].fillna(df["Marks"].mean())
print("clean df:\n",df)
print("Missing values:",df.isnull().sum().sum())