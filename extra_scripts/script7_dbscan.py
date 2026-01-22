import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#Configuration
TFIDF_PATH = "./data/tfidf_vectors.csv"
ISSUES_PATH = "./data/issues.csv"
OUTPUT_ASSIGNMENTS = "./data/cluster_assignments.csv"
OUTPUT_CENTERS = "./data/cluster_centers.csv"
OUTPUT_PLOT = "./data/severity_score_elbow.png"
OUTPUT_DIR = "./data/cluster_histograms"

EPSILON = 0.55  #DBSCAN eps parameter
MIN_SAMPLES = 10  #DBSCAN min_samples parameter
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
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f"Saved elbow plot to {OUTPUT_PLOT}")

#STEP 2: Filter vectors by threshold and cluster (from script2_dbscan.py)

#Filter vectors with score > threshold
mask = distances > threshold
filtered_vectors = vectors_array[mask]
filtered_original_indices = np.where(mask)[0]

print(f"Filtered to {len(filtered_vectors)} issues with severity score > {threshold:.2f}")

#Run DBSCAN clustering on filtered vectors
dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric="cosine")
labels = dbscan.fit_predict(filtered_vectors)

#Save cluster assignments with original row IDs
assignments = pd.DataFrame({
    "row_id": filtered_original_indices,
    "cluster": labels
})
assignments.to_csv(OUTPUT_ASSIGNMENTS, index=False)

print(f"Saved cluster assignments to {OUTPUT_ASSIGNMENTS}")

#Calculate the centers of each cluster
unique_labels = set(labels)
cluster_centers = []
for label in sorted(unique_labels):
    if label == -1:
        continue  #Skip noise points
    cluster_points = filtered_vectors[labels == label]
    center = np.mean(cluster_points, axis=0)
    cluster_centers.append(center)

#Save cluster centers
centers_df = pd.DataFrame(
    cluster_centers,
    columns=vectors_df.columns
)
centers_df.to_csv(OUTPUT_CENTERS, index=False)

print(f"Saved cluster centers to {OUTPUT_CENTERS}")

#Display number of issues for each cluster
cluster_counts = pd.Series(labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    if cluster_id == -1:
        print(f"Noise points: {count} issues")
    else:
        print(f"Cluster {cluster_id}: {count} issues")

#STEP 3: Generate histograms (from script3_dbscan.py)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for cluster_idx, row in centers_df.iterrows():
    #Sort words by weight
    sorted_words = row.sort_values(ascending=False).head(TOP_N_WORDS)

    plt.figure(figsize=(10, 5))
    plt.bar(sorted_words.index, sorted_words.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Center TF-IDF Weight")
    plt.title(f"Cluster {cluster_idx}: Top {TOP_N_WORDS} Words")

    plt.tight_layout()
    output_path = os.path.join(
        OUTPUT_DIR, f"cluster_{cluster_idx}_top_words.png"
    )
    plt.savefig(output_path)
    plt.close()

    print(f"Saved histogram for cluster {cluster_idx}: {output_path}")