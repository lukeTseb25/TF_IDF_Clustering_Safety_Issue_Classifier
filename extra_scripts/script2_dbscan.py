import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#Configuration
TFIDF_PATH = "./data/tfidf_vectors.csv"
OUTPUT_ASSIGNMENTS = "./data/cluster_assignments.csv"
OUTPUT_CENTERS = "./data/cluster_centers.csv"
EPSILON = 0.3 #DBSCAN eps parameter
MIN_SAMPLES = 10 #DBSCAN min_samples parameter

#Load data
X = pd.read_csv(TFIDF_PATH)
X_np = X.values

#Final clustering
dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric="cosine")
labels = dbscan.fit_predict(X_np)

#Save outputs
assignments = pd.DataFrame({
    "row_id": np.arange(len(labels)),
    "cluster": labels
})
assignments.to_csv(OUTPUT_ASSIGNMENTS, index=False)

#Calculate the centers of each cluster
unique_labels = set(labels)
cluster_centers = []
for label in unique_labels:
    if label == -1:
        continue  # Skip noise points
    cluster_points = X_np[labels == label]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

#Save cluster centers
centers = pd.DataFrame(
    cluster_centers,
    columns=X.columns
)
centers.to_csv(OUTPUT_CENTERS, index=False)

#Display number of issues for each cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} issues")