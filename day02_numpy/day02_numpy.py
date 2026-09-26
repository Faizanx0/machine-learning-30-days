import numpy as np

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




