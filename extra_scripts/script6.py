import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_PLOT = "./data/severity_score_elbow.png"

def detect_elbow(y):
    sorted = np.sort(y)
    x = np.arange(len(sorted))

    p1 = np.array([x[0], sorted[0]])
    p2 = np.array([x[-1], sorted[-1]])

    distances = np.abs(
        np.cross(p2 - p1, p1 - np.column_stack((x, sorted)))
    ) / np.linalg.norm(p2 - p1)

    elbow_idx = np.argmax(distances)
    
    return elbow_idx, x, sorted

# Read the vectors and issues
vectors_df = pd.read_csv('data/tfidf_vectors.csv')
issues_df = pd.read_csv('data/issues.csv')

vectors_array = np.array(vectors_df.values)
distances = vectors_array.sum(axis=1)

# Detect elbow
elbow_idx, x, y = detect_elbow(distances)
threshold = y[elbow_idx]

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