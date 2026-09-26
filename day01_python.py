def analyze_marks(marks):
    print("Total:",sum(marks))
    print("Count:",len(marks))
    print("Average:",sum(marks)/len(marks))
    print("Max:",max(marks))
    print("Min:",min(marks))
    [print(mark) for mark in marks if mark>=80]

Student_marks=[72, 85, 91, 64, 78, 88, 55, 93]
analyze_marks(Student_marks)
