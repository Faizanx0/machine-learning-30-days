"""
1. Run the model
Print the DataFrame with the Cluster column. Check that every student has a cluster ID.

2. Count students
Use value_counts() on df['Cluster'] to find how many students are in each group.

3. Understand the groups
Use groupby('Cluster') to calculate the average StudyHours and PreviousMarks for each cluster.

4. Experiment with K
Change n_clusters from 3 to 2. Run it again and compare the group counts with your first result. - First the ans 0-6,1-9,2-5 but after changing to 3 to 2 the value become 1-11,0-9 so the overall value are 20 but each change its cluster acc to thier type

5. Explain the concepts
In your own words, explain unsupervised learning, centroid, and why we scale features before K-Means. - Unsupervised learning in which the data doesnot have the answer,so the model make eaxh one into group called cluster each cluter have its mid point called centroid, we scale feature beacuse StandardScaler() helps the data to come in equvalent its number btw -1to1 this helps data to under each one value
"""

import pandas as pd
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
X = df[["StudyHours","PreviousMarks"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init = 10
    )

model.fit(X_scaled)

df["Cluster"] = model.labels_
print("Data:",df.sort_values(by="Cluster"))
print("Count:\n",df["Cluster"].value_counts())
print("Avg:\n",df.groupby("Cluster")[["StudyHours","PreviousMarks"]].mean())
