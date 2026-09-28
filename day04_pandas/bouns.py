"""Find how many students are in each department, then calculate the average marks for CSE students only.

Hint: use value_counts() for the first part and filtering plus mean() for the second."""

import pandas as pd

data = {
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir", "Noor"],
    "Age": [19, 20, 19, 21, 20, 19],
    "Marks": [85, 92, 67, 78, 95, 73],
    "Department": ["CSE", "IT", "CSE", "IT", "CSE", "IT"]
}

df = pd.DataFrame(data)
print("Count",df["Department"].value_counts())
print("Average CSE:",df[df["Department"]=="CSE"]["Marks"].mean())