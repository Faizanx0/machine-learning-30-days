"""1. Find missing values
Display True/False for missing values using isnull().
2. Count missing values
Count missing values in each column, then calculate the total across the dataset.
3. Display incomplete rows
Show all rows containing at least one missing value.
4. Remove missing marks
Create a new DataFrame by removing rows where Marks is missing. Keep the original df unchanged.
5. Fill missing ages
Replace missing ages with the median of the available ages.
6. Fill missing marks
Replace missing marks with the mean of the available marks.
7. Find duplicates
Detect exact duplicate rows and count them. Then remove exact duplicate rows.
8. Clean department names
Remove extra spaces and convert Department values to uppercase."""

import pandas as pd
import numpy as np

data = {
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir", "Sara"],
    "Age": [19, 20, np.nan, 21, 20, 20],
    "Marks": [85, np.nan, 67, 78, 95, np.nan],
    "Department": ["CSE", "IT", "CSE", "IT", "CSE", "IT"]
}

df = pd.DataFrame(data)
print("Missing:\n",df.isnull())
print("Count missing:\n",df.isnull().sum())
print("Total missing:",df.isnull().sum().sum())
print("Incomplete row:\n",df[df.isnull().any(axis=1)])
marks_df=df.dropna(subset=["Marks"])
print("Remove Missing marks:\n",marks_df)
df["Age"]=df["Age"].fillna(df["Age"].median())
print("Fill Age:\n",df)
df["Marks"]=df["Marks"].fillna(df["Marks"].mean())
print("Fill Marks:\n",df)
print("Duplicate:\n",df.duplicated().sum())
df=df.drop_duplicates()
print("Remove duplicate:\n",df)
df["Department"]=df["Department"].str.strip().str.upper()
print("Clean df:\n",df)