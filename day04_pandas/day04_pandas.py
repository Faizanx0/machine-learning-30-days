
"""1. Inspect the dataset
Print the first 3 rows, shape, column names, and data types.
2. Select columns
Print only the Name column, then print Name and Marks together.
3. Select rows
Print the first row and the first three rows using iloc.
4. Filter marks
Display all students who scored at least 80.
5. Filter by department
Display only students whose Department is "CSE".
6. Combine conditions
Display students who scored at least 80 AND are 19 years old.
7. Calculate statistics
Find the mean, median, highest, and lowest marks.
8. Find the topper
Display the complete row of the student with the highest marks.
9. Count departments
Count how many students are in each department using value_counts().
10. Create a CSV
Save the DataFrame to "students.csv" without saving the index."""

import pandas as pd

data = {
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir", "Noor"],
    "Age": [19, 20, 19, 21, 20, 19],
    "Marks": [85, 92, 67, 78, 95, 73],
    "Department": ["CSE", "IT", "CSE", "IT", "CSE", "IT"]
}

df = pd.DataFrame(data)
print("First 3 rows:\n",df[:3])
print("Shape:",df.shape)
print("Column Name:",df.columns)
print("Data Type:\n",df.dtypes)
print("Name Col:\n",df["Name"])
print("Name Marks Col:\n",df[["Name","Marks"]])
print("First Row:\n",df.iloc[:1])
print("First 3 Row:\n",df.iloc[:3])
print("Marks >= 80:\n",df[df.Marks>=80])
print("Department:\n",df[df.Department=="CSE"])
print("Marks 80 & Age 19:\n",df[(df.Marks>=80)&(df.Age==19)])
print("Mean:",df["Marks"].mean())
print("Median:",df["Marks"].median())
print("Highest:",df["Marks"].max())
print("Lowest:",df["Marks"].min())
print("Topper:\n",df.loc[df["Marks"].idxmax()])
print("Count Dept:\n",df.value_counts("Department"))
print("Filter\n",df.value_counts("Department"))
df.to_csv("students.csv",index=False)
