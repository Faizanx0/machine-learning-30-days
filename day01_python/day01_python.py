"""Task 1
Print the total marks.
Task 2
Print how many marks are present.
Task 3
Calculate and print the average.
Task 4
Print the highest mark.
Task 5
Print the lowest mark.
Task 6
Use a loop to print only the marks that are 80 or above.
Task 7
Create a function: def analyze_marks(marks):
Total: ...
Count: ...
Average: ...
Highest: ...
Lowest: ..."""

def analyze_marks(marks):
    print("Total:",sum(marks))
    print("Count:",len(marks))
    print("Average:",sum(marks)/len(marks))
    print("Max:",max(marks))
    print("Min:",min(marks))
    [print(mark) for mark in marks if mark>=80]

Student_marks=[72, 85, 91, 64, 78, 88, 55, 93]
analyze_marks(Student_marks)
