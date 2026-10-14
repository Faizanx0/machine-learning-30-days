"""
1. Run the code
Share the inertia values printed for K = 1 through 8.

2. Read the graph
Which K looks like the elbow in your actual graph? Explain why you chose it. - I think 2 becoz by seen the graph and the sudden downfall of val leads to make 2 as elbow

3. Understand inertia
Why does inertia generally decrease when K increases? Why shouldn't we automatically choose the largest K? - When k inc the no of groups inc the formula of inertia means the heavier the mas the more inertia so if less no of feature in cluster becoz the no of cluster inc, we cant choose biggger k becoz the problem sometime we need ans like yes or no so the many clusters the many poosibly of ans but we wnt smetimes

4. Experiment
Change n_init from 10 to 1, run the experiment again, and compare the results. Explain why the inertia values might change. - I reverse the n_init but value of each k not changed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "StudyHours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 7, 3, 9, 5, 8, 3, 6, 9, 4
    ],
    "PreviousMarks": [
        35, 40, 45, 50, 55, 60, 65, 80, 42, 52,
        68, 75, 46, 88, 58, 78, 48, 67, 92, 54
    ]
}

df = pd.DataFrame(data)

X = df[["StudyHours", "PreviousMarks"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
k_value = range(1,9)

for k in k_value:
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

print("K Value:",list(k_value))
print("Inertia",np.round(inertia,2))
plt.plot(list(k_value), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(list(k_value))
plt.grid(True)
plt.show()