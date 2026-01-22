import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Configuration
TFIDF_PATH = "./data/tfidf_vectors.csv"
OUTPUT_ASSIGNMENTS = "./data/cluster_assignments.csv"
OUTPUT_CENTROIDS = "./data/cluster_centroids.csv"
OUTPUT_PLOT = "./data/kmeans_elbow.png"

MAX_K = 20
ELBOW_PATIENCE = 3
ALLOW_CLUSTER_CHOICE = False
MANUAL_K = 6  # used only if ALLOW_CLUSTER_CHOICE = True

#Load data
X = pd.read_csv(TFIDF_PATH)
X_np = X.values

#Manual override
if ALLOW_CLUSTER_CHOICE:
    best_k = MANUAL_K
else:
    inertias = []
    elbow_k = None
    remaining_patience = ELBOW_PATIENCE

    for k in range(2, MAX_K):
        km = KMeans(n_clusters=k, random_state=777, n_init=10)
        km.fit(X_np)
        inertias.append(km.inertia_)

        p1 = np.array([0, inertias[0]])
        p2 = np.array([len(inertias) - 1, inertias[-1]])

        distances = np.abs(np.cross(p2 - p1, p1 - np.column_stack((np.arange(len(inertias)), inertias))))/np.linalg.norm(p2 - p1)
        elbow_idx = np.argmax(distances)
        if elbow_k is None or elbow_idx != elbow_k:
            elbow_k = elbow_idx + 2
            remaining_patience = ELBOW_PATIENCE
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    
    best_k = elbow_k if elbow_k else MAX_K

#Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=777, n_init=20)
labels = kmeans.fit_predict(X_np)

#Save outputs
assignments = pd.DataFrame({
    "row_id": np.arange(len(labels)),
    "cluster": labels
})
assignments.to_csv(OUTPUT_ASSIGNMENTS, index=False)

centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
)
centroids.to_csv(OUTPUT_CENTROIDS, index=False)

#Plot elbow curve
if not ALLOW_CLUSTER_CHOICE:
	plt.figure()
	plt.plot(range(2, 2 + len(inertias)), inertias)
	plt.axvline(best_k, linestyle="--", label=f"k={best_k}")
	plt.xlabel("k")
	plt.ylabel("Inertia")
	plt.title("K-Means Elbow")
	plt.legend()
	plt.tight_layout()
	plt.savefig(OUTPUT_PLOT)
	plt.close()

	print(f"Selected k = {best_k}")

#Display number of issues for each cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} issues")