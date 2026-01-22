import os
import pandas as pd
import matplotlib.pyplot as plt

#Configuration
CENTERS_PATH = "./data/cluster_centers.csv"
OUTPUT_DIR = "./data/cluster_histograms"

TOP_N_WORDS = 20  # number of most important words per cluster

#Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

centers = pd.read_csv(CENTERS_PATH)

#Generate histograms
for cluster_idx, row in centers.iterrows():
    # Sort words by weight
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
