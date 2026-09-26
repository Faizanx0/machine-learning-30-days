"""Create a two-dimensional NumPy array containing marks for 3 students across 3 subjects.

Then:

Print the shape of the array.

Print the marks of the second student.

Print the mark in the third row and first column.

Try this only after finishing the main exercise."""

import numpy as np

student=np.array([
    [88,75,95],
    [60,85,75],
    [90,92,97]
    ])
print(student.shape)
print("Second Student:",student[1])
print("(3,1)th marks:",student[2][0])