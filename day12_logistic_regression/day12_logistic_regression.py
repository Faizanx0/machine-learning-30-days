"""
Exercise 1 — Identify TP/TN/FP/FN
Given:
actual =    [1, 1, 0, 0, 1, 0, 1, 0]
predicted = [1, 0, 0, 1, 1, 1, 1, 0]
Manually find:
TP = ?
TN = ?
FP = ?
FN = ?

Don't use sklearn for this one.
Exercise 2 — Calculate manually
Using your TP/TN/FP/FN:
Calculate:
Accuracy
Precision
Recall
F1 Score

Try the formulas yourself.
Exercise 3 — Use sklearn
Use:
from sklearn.metrics import (    confusion_matrix,    accuracy_score,    precision_score,    recall_score,    f1_score)


Calculate all four metrics.
Exercise 4 — Explain in your own words
Answer:
1. What is a True Positive? - It means that how many true prediction are Positive
2. What is a False Positive? - It means that how many False prediction are Positive
3. What is a False Negative? - It means that how many False prediction are Negative
4. Difference between Precision and Recall? - Precision means that how many true Precision are actually true; Recall means that how many trues are actually present in my Precision
5. Why can accuracy sometimes be misleading? - Accuracy might be misleading becoz it does not means how you model is accurate it tell how many true prediction on overall prediction
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

actual = np.array([1, 1, 0, 0, 1, 0, 1, 0])
predicted = np.array([1, 0, 0, 0, 1, 1, 1, 0])

cm = confusion_matrix(actual,predicted)
TP=cm[0][0]
TN=cm[1][1]
FP=cm[1][0]
FN=cm[0][1]
print("Confusion Matrix:\n",cm)
print("TP:",TP)
print("TN",TN)
print("FP",FP)
print("FN",FN)

accuracy = (TP+TN)/(TP+TN+FP+FN)
print("Accuracy:",accuracy)
precision = (TP)/(TP+FP)
print("Precision",precision)
recall = (TP)/(TP+FN)
print("Recall:",recall)
f1 = 2*(precision*recall)/(precision+recall)
print("F1 Score:",f1)

print("Using sklearn:",
        accuracy_score(actual,predicted),
        precision_score(actual,predicted),
        recall_score(actual,predicted),
        f1_score(actual,predicted)
    )