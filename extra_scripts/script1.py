import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

#Configuration
CSV_PATH = "./data/issues.csv"
OUTPUT_TFIDF = "./data/tfidf_vectors.csv"
OUTPUT_PLOT = "./data/idf_elbow.png"

WORD_REGEX = re.compile(r"(?<![0-9A-Za-z])[A-Za-z]+(?![0-9A-Za-z])")

#Tokenizer
def tokenizer(text):
    return WORD_REGEX.findall(text.lower())

#Load data
texts = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        texts.append(
            (row.get("issue_title", "") or "") + " " +
            (row.get("issue_body", "") or "")
        )

#TF-IDF
vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    lowercase=False,
    token_pattern=None,
    norm="l2"
)

X = vectorizer.fit_transform(texts)
idf = vectorizer.idf_
vocab = vectorizer.get_feature_names_out()

#Elbow detection (max distance)
idf_sorted = np.sort(idf)
x = np.arange(len(idf_sorted))

p1 = np.array([x[0], idf_sorted[0]])
p2 = np.array([x[-1], idf_sorted[-1]])

distances = np.abs(
    np.cross(p2 - p1, p1 - np.column_stack((x, idf_sorted)))
) / np.linalg.norm(p2 - p1)

elbow_idx = np.argmax(distances)
idf_threshold = idf_sorted[elbow_idx]

print(f"Selected IDF threshold: {idf_threshold:.4f}")

#Plot elbow
plt.figure()
plt.plot(idf_sorted)
plt.axvline(elbow_idx, linestyle="--", label="Elbow")
plt.axhline(idf_threshold, linestyle=":", label=f"IDF={idf_threshold:.2f}")
plt.xlabel("Words (sorted by IDF)")
plt.ylabel("IDF")
plt.title("IDF Elbow Cutoff")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

#Filter vocabulary
#Document frequency per word
df_counts = (X > 0).sum(axis=0).A1
n_docs = X.shape[0]

#Tunable bounds
MIN_DF = max(2, int(0.005 * n_docs)) #appears in at least 0.5% of (2) issues
MAX_DF = int(1.0 * n_docs)

print(f"min_df = {MIN_DF} documents")
print(f"max_df = {MAX_DF} documents")

keep_words = []

for word, idf_val, df_val in zip(vocab, idf, df_counts):
    if (
        idf_val >= idf_threshold and
        df_val >= MIN_DF and
        df_val <= MAX_DF
    ):
        keep_words.append(word)

print(f"Kept {len(keep_words)} words after balanced filtering")

filtered_vectorizer = TfidfVectorizer(
    tokenizer=tokenizer,
    lowercase=False,
    token_pattern=None,
    norm="l2",
    vocabulary=keep_words
)

X_filtered = filtered_vectorizer.fit_transform(texts)

df = pd.DataFrame(
    X_filtered.toarray(),
    columns=filtered_vectorizer.get_feature_names_out()
)

df.to_csv(OUTPUT_TFIDF, index=False)

print(f"Saved TF-IDF matrix: {OUTPUT_TFIDF}")
print(f"Vocabulary size: {df.shape[1]}")
