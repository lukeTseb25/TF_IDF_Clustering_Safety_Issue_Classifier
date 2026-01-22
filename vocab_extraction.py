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

WORD_REGEX = re.compile(r"accident[a-z]*|admin[a-z]*|apikey|audit[a-z]*|auth[a-z]*|breach[a-z]*|buffer[a-z]*|bypass[a-z]*|byte|chmod|clobber[a-z]*|complian[a-z]+|concern[a-z]*|concurren[a-z]+|condition|config[a-z]*|confirm[a-z]*|consent[a-z]*|console|constrain[a-z]*|consum[a-z]+|conver(?!sat)[a-z]+|corrupt[a-z]*|cpu|crash[a-z]*|credential|critical|\bdead\b|deadlock[a-z]*|decod[a-z]+|delet[a-z]+|denial|destr[a-z]+|disclos[a-z]+|disrupt[a-z]*|drop[a-z]*|dump[a-z]*|duplicat[a-z]+|edit(?!or)|eliminat[a-z]+|encod[a-z]+|encrypt[a-z]*|enforc[a-z]+|eras[a-z]+|escap[a-z]+|except[a-z]*|exclud[a-z]+|expan[a-z]+|expos[a-z]+|extract[a-z]*|fail[a-z]*|fault|forbid[a-z]*|freez[a-z]+|hash[a-z]*|heap|illegal[a-z]*|inconsisten[a-z]+|inject[a-z]*|interrupt[a-z]*|kernel|key|leak[a-z]*|lock[a-z]*|log[a-z]*|loop[a-z]*|loss|memory|modif[a-z]+|\bmv\b|null|oauth|overhead|overload[a-z]*|overrid[a-z]+|overwrit[a-z]+|password|persist|permission|perms|personal|plaintext|polic[a-z]+|privelege[a-z]*|protect[a-z]*|\brace|raise|\bram\b|recover[a-z]*|redact[a-z]*|remov[a-z]+|replac[a-z]+|reset|resource|revert|revers[a-z]+|\brm\b|safety|salt(?!y)|sanatiz[a-z]+|secret[a-z]*|segfault|sensitive|\bsets\b|sever[a-z]+|spawned|spin[a-z]*|stack|stall|undo|undocumented|unencrypt|uninstall|violat[a-z]+|wipe")

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

idf_sorted = np.sort(idf)

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
