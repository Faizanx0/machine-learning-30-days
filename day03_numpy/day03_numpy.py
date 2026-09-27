import numpy as np

marks = np.array([
    [72, 85, 91],
    [64, 78, 88],
    [55, 93, 67],
    [81, 76, 90]
])

"""Exercise checklist
1. Print the shape, number of dimensions and total number of elements.
2. Print the marks of the second student.
3. Print the ML mark of the fourth student.
4. Print the Maths marks of all students.
5. Print the first two students and their first two subject marks.
6. Add 5 bonus marks to every element and print the new array.
7. Calculate the total marks for each student using axis.
8. Calculate the average marks for each subject using axis.
9. Find the highest mark in each subject using axis.
10. Filter and print all individual marks greater than or equal to 80.
Each row represents a student. The columns represent Python, Maths and ML marks, respectively."""

print("Shape:",marks.shape)
print("Dimension:",marks.ndim)
print("Total Element:",marks.size)
print("2nd Student marks:",marks[1,:])
print("ML marks:",marks[3][2])
print("Math marks:",marks[:,1])
print("First Two:\n",marks[:2,:2])
print("Add 5:\n",marks+5)
print("Total marks:",np.sum(marks,axis=1))
print("Average marks:",np.mean(marks,axis=0))
print("Highest marks:",np.max(marks,axis=0))
print("Filter:",marks[marks>=80]) 