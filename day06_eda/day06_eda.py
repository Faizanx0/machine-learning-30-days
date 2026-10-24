"""1. Explore the dataset
Print the shape, column names, and numerical summary using describe().

2. Calculate total marks
Create a Total column by adding Maths, Science, and English.

3. Calculate average marks
Create an Average column for each student.

4. Filter students
Display students whose average is at least 85.

5. Sort students
Sort the DataFrame by Total from highest to lowest and show the top 3.

6. Department analysis
Use groupby() to calculate the average Total for each department.

7. Subject comparison
Calculate the class average for each subject and find the subject with the highest average.

8. Department summary
Use groupby() and agg() to show count, mean, minimum, and maximum Total for each department.

9. Find the topper
Display the full row of the student with the highest Total.
"""

import pandas as pd
import numpy as np

data = {
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir",
             "Noor", "Isha", "Arjun", "Meera", "Ali"],
    "Department": ["CSE", "IT", "CSE", "IT", "CSE",
                   "IT", "CSE", "IT", "CSE", "IT"],
    "Maths": [85, 92, 67, 78, 95, 73, 88, 81, 76, 90],
    "Science": [88, 89, 72, 84, 91, 79, 86, 75, 82, 93],
    "English": [78, 94, 70, 80, 87, 85, 90, 79, 83, 88]
}

df = pd.DataFrame(data)

print("Shape:",df.shape)
print("Column:",df.columns)
print("Summary:\n",df.describe())
df["Total"]=(df["Maths"]+df["Science"]+df["English"])
df["Average"]=(df["Total"]/3)
print("Avg >= 85:\n",df[df["Average"]>=85])
topper=df.sort_values(
        by="Total",
        ascending=False
    )
print("Top 3:\n",topper.head(3))
avg_dept=df.groupby("Department")["Total"].mean()
print("Avg Dept:\n",avg_dept)
subject_averages = df[["Maths", "Science", "English"]].mean()
print("Subject Avg:",subject_averages.idxmax(),subject_averages.max())
dept_summary=df.groupby("Department")["Total"].agg(
    ["count","mean","min","max"])
print("Dept Summary:\n",dept_summary)
print("Topper:\n",topper.head(1))

df.to_csv("student_performance_analysis.csv", index=False)