import os
from tkinter import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Configuration
TFIDF_PATH = "./data/tfidf_vectors.csv"
ISSUES_PATH = "./data/issues.csv"
OUTPUT_ASSIGNMENTS = "./data/cluster_assignments.csv"
OUTPUT_CENTROIDS = "./data/cluster_centroids.csv"
KMEANS_PLOT = "./data/kmeans_elbow.png"
SEVERITY_PLOT = "./data/severity_score_elbow.png"
OUTPUT_DIR = "./data/cluster_histograms"

MAX_K = 20
ELBOW_PATIENCE = 3
ALLOW_CLUSTER_CHOICE = True
MANUAL_K = 5  # used only if ALLOW_CLUSTER_CHOICE = True
TOP_N_WORDS = 20  #number of most important words per cluster

#STEP 1: Calculate severity scores and detect elbow (from script6.py)

def detect_elbow(y):
    sorted_y = np.sort(y)
    x = np.arange(len(sorted_y))

    p1 = np.array([x[0], sorted_y[0]])
    p2 = np.array([x[-1], sorted_y[-1]])

    distances = np.abs(
        np.cross(p2 - p1, p1 - np.column_stack((x, sorted_y)))
    ) / np.linalg.norm(p2 - p1)

    elbow_idx = np.argmax(distances)
    
    return elbow_idx, x, sorted_y

#Read the vectors and issues
vectors_df = pd.read_csv(TFIDF_PATH)
issues_df = pd.read_csv(ISSUES_PATH)

vectors_array = np.array(vectors_df.values)
distances = vectors_array.sum(axis=1)

#Detect elbow
elbow_idx, x, y = detect_elbow(distances)
threshold = y[elbow_idx]

print(f"Elbow detected at index {elbow_idx} with threshold {threshold:.2f}")

#Plot elbow
plt.figure()
plt.plot(y)
plt.axvline(elbow_idx, linestyle="--", label="Elbow")
plt.axhline(threshold, linestyle=":", label=f"Score={threshold:.2f}")
plt.xlabel("Issues")
plt.ylabel("Severity Score")
plt.title("Elbow Cutoff")
plt.legend()
plt.tight_layout()
plt.savefig(SEVERITY_PLOT)
plt.close()

print(f"Saved elbow plot to {SEVERITY_PLOT}")

#STEP 2: Filter vectors by threshold and cluster (from script2_kmeans.py)

#Filter vectors with score > threshold
mask = distances > threshold
filtered_vectors = vectors_array[mask]
filtered_original_indices = np.where(mask)[0]

print(f"Filtered to {len(filtered_vectors)} issues with severity score > {threshold}")

#Run KMeans clustering on filtered vectors using elbow method
if ALLOW_CLUSTER_CHOICE:
    best_k = MANUAL_K
else:
    inertias = []
    elbow_k = None
    remaining_patience = ELBOW_PATIENCE

    for k in range(2, MAX_K):
        km = KMeans(n_clusters=k, random_state=777, n_init=10)
        km.fit(filtered_vectors)
        inertias.append(km.inertia_)

        p1 = np.array([0, inertias[0]])
        p2 = np.array([len(inertias) - 1, inertias[-1]])

        distances_elbow = np.abs(np.cross(p2 - p1, p1 - np.column_stack((np.arange(len(inertias)), inertias))))/np.linalg.norm(p2 - p1)
        elbow_idx = np.argmax(distances_elbow)
        if elbow_k is None or elbow_idx != elbow_k:
            elbow_k = elbow_idx + 2
            remaining_patience = ELBOW_PATIENCE
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    
    best_k = elbow_k if elbow_k else MAX_K

if not ALLOW_CLUSTER_CHOICE:
	plt.figure()
	plt.plot(range(2, 2 + len(inertias)), inertias)
	plt.axvline(best_k, linestyle="--", label=f"k={best_k}")
	plt.xlabel("k")
	plt.ylabel("Inertia")
	plt.title("K-Means Elbow")
	plt.legend()
	plt.tight_layout()
	plt.savefig(KMEANS_PLOT)
	plt.close()

	print(f"Selected k = {best_k}")

#Final KMeans clustering
kmeans = KMeans(n_clusters=best_k, random_state=777, n_init=20)
labels = kmeans.fit_predict(filtered_vectors)

#Save cluster assignments with original row IDs and severity scores
filtered_severity_scores = distances[mask]
assignments = pd.DataFrame({
    "row_id": filtered_original_indices,
    "severity_score": filtered_severity_scores,
    "cluster": labels
})
assignments.to_csv(OUTPUT_ASSIGNMENTS, index=False)

print(f"Saved cluster assignments to {OUTPUT_ASSIGNMENTS}")

#Calculate the centroids of each cluster
centroids_list = kmeans.cluster_centers_

#Save cluster centroids
centroids_df = pd.DataFrame(
    centroids_list,
    columns=vectors_df.columns
)
centroids_df.to_csv(OUTPUT_CENTROIDS, index=False)

print(f"Saved cluster centroids to {OUTPUT_CENTROIDS}")

#Display number of issues for each cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id}: {count} issues")

#STEP 3: Generate histograms (from script3_kmeans.py)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for cluster_idx, row in centroids_df.iterrows():
    #Sort words by weight
    sorted_words = row.sort_values(ascending=False).head(TOP_N_WORDS)

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_words.index, sorted_words.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Centroid TF-IDF Weight")
    plt.title(f"Cluster {cluster_idx}: Top {TOP_N_WORDS} Words")

    plt.tight_layout()
    output_path = os.path.join(
        OUTPUT_DIR, f"cluster_{cluster_idx}_top_words.png"
    )
    plt.savefig(output_path)
    plt.close()

    print(f"Saved histogram for cluster {cluster_idx}: {output_path}")