import numpy as np
"""
Exercise checklist
10 tasks
1. Print the entire NumPy array.
2. Print the number of elements using shape.
3. Calculate the total marks using NumPy.
4. Calculate the mean (average).
5. Find the median.
6. Find the highest and lowest marks.
7. Calculate the standard deviation.
8. Print the first 5 marks using slicing.
9. Print only the marks greater than or equal to 80 using Boolean indexing.
10. Multiply every mark by 2 and print the resulting array."""

marks = np.array([72, 85, 91, 64, 78, 88, 55, 93, 67, 81])
print("Marks:",marks)
print("Length:",marks.shape)
print("Total:",np.sum(marks))
print("Mean:",np.mean(marks))
print("Median:",np.median(marks))
print("Highest:",np.max(marks))
print("Lowest:",np.min(marks))
print("Standard deviation: %.2f"%np.std(marks))
print("First 5:",marks[0:5])
print("Greater than 80:",marks[marks>=80])
print("Multiply by 2:",marks*2)




