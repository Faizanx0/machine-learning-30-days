"""Exercise 1 — Python basics
Create a list called marks containing 65, 82, 91, 74, 88, and 59. Write a function named analyze_marks(marks) that returns the average. Then use a loop to print every mark greater than or equal to 80.
Exercise 2 — NumPy arrays
Create a 2D NumPy array with rows [70, 80, 90], [60, 75, 85], and [95, 88, 92]. Print its shape, the second row, the third column, and all values greater than or equal to 85.
Exercise 3 — NumPy axis
Using the same array, calculate the total marks of each student and the total marks of each subject. Also calculate the average of each student. Decide whether each operation needs axis=0 or axis=1.
Exercise 4 — Pandas filtering
Create a DataFrame with Name = [Aman, Sara, Ravi, Zoya, Kabir], Department = [CSE, IT, CSE, IT, CSE], and Marks = [85, 92, 67, 78, 95]. Display only Name and Marks, students with Marks >= 80, and CSE students with Marks >= 80.
Exercise 5 — Missing values
Create a DataFrame with Name = [Aman, Sara, Ravi, Zoya], Age = [19, NaN, 20, 21], and Marks = [85, 92, NaN, 78]. Count missing values, fill Age using its median, fill Marks using its mean, and count duplicate rows. Import NumPy as np to use np.nan.
Exercise 6 — Grouping
Using the DataFrame from Exercise 4, calculate the average Marks for each Department. Then show the count, mean, minimum and maximum Marks for each department using agg().
Exercise 7 — Sorting
Sort the students from Exercise 4 by Marks from highest to lowest. Display the top two students and retrieve the complete row of the student with the highest marks using idxmax() and loc[].
Exercise 8 — Mini-project
Recreate your Day 6 student performance project with Name, Department, Maths, Science and English. Calculate each student's Total and Average, display students with Average >= 85, show the top 3 students, calculate department-wise average Total, identify the subject with the highest average, and export the final DataFrame to CSV.
"""

import numpy as np
import pandas as pd
marks=[65,82,91,74,88,59]
def analyze_marks(marks):
    return sum(marks)/len(marks)
print("Average Marks:",analyze_marks(marks))
x=[mark for mark in marks if mark>=80]
print("Marks >= 80:\n",x)

arr=np.array([
    [70, 80, 90],
    [60, 75, 85],
    [95, 88, 92]
    ])
print("Shape:",arr.shape)
print("2nd Row:",arr[1])
print("3rd Col:",arr[:,2])
print("Marks >= 85:",arr[arr>=85])

print("Total Marks each students:",np.sum(arr,axis=1))
print("Total Marks each subject:",np.sum(arr,axis=0))
print("Average Marks each students:",np.round(np.mean(arr,axis=1),2))

df=pd.DataFrame({
    "Name" : ["Aman", "Sara", "Ravi", "Zoya", "Kabir"], 
    "Department" : ["CSE", "IT", "CSE", "IT", "CSE"],  
    "Marks" : [85, 92, 67, 78, 95]
})
print("Students:\n",df[["Name","Marks"]])
print("Students Marks >= 80:\n",df[df["Marks"]>=80])
print("Students Marks >= 80 And CSE Dept:\n",df[(df["Department"]=="CSE") & (df["Marks"]>=80)])

data=pd.DataFrame({
    "Name" : ["Aman", "Sara", "Ravi", "Zoya"],
    "Age" : [19, np.nan, 20, 21],
    "Marks" : [85, 92,  np.nan, 78]
})

print("Missing Value:\n",data.isnull().sum())
data["Age"]=data["Age"].fillna(data["Age"].median())
data["Marks"]=data["Marks"].fillna(data["Marks"].mean())
print("Filled data:\n",data)
print("Duplicate :",data.duplicated().sum())

print("Average Marks each Dept:\n",df.groupby("Department")["Marks"].mean())
print("Department info:\n",df.groupby("Department")["Marks"].agg(["count","mean","min","max"]))

topper=data.sort_values(
    by="Marks",
    ascending=False
    )
print("Top 2:\n",topper.head(2))
print("Topper:\n",data.loc[data["Marks"].idxmax()])

Class = pd.DataFrame({
    "Name": ["Aman", "Sara", "Ravi", "Zoya", "Kabir",
             "Noor", "Isha", "Arjun", "Meera", "Ali"],
    "Department": ["CSE", "IT", "CSE", "IT", "CSE",
                   "IT", "CSE", "IT", "CSE", "IT"],
    "Maths": [85, 92, 67, 78, 95, 73, 88, 81, 76, 90],
    "Science": [88, 89, 72, 84, 91, 79, 86, 75, 82, 93],
    "English": [78, 94, 70, 80, 87, 85, 90, 79, 83, 88]
})

Class["Total"]=Class["Maths"]+Class["Science"]+Class["English"]
Class["Average"]=np.round(Class["Total"]/3,2)

print("Avg >= 85:\n",Class[Class["Average"]>=85])
print("Top 3:\n",Class.sort_values(
    by="Total",
    ascending=False
    ).head(3))
print("Dept Avg:\n",Class.groupby("Department")["Total"].mean())
Subject_Avg=Class[["Maths","Science","English"]].mean()
print("Highest Subject Avg:",Subject_Avg.idxmax(),Subject_Avg.loc[Subject_Avg.idxmax()])
print("Lowest Total marks:\n",Class.loc[Class["Total"].idxmin()])
Class.to_csv("Mini-Project.csv",index=False)
