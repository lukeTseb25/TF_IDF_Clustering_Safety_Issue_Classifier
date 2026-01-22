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

WORD_REGEX = re.compile(r"abort|accept|access|accident|acknowledge|alloc|anonym|apikey|api|assert|assign|attributes|audit|auth|availability|backup|block|boolean|bot|boundar|breach|buffer|bypass|byte|care|cargo|checkout|chmod|clobber|cloud|compile|complian|concern|concurren|condition|config|confirm|consent|console|constrain|consume|conver|convert|cookie|corrupt|cpu|crash|credential|criteria|dead|deadlock|decod|delete|denail|depend|deploy|destr|develop|diff|disclos|disk|disrupt|distinct|domain|downgrad|download|drive|drop|dump|duplicate|edited|eliminate|encod|encrypt|enforce|enhance|enhancing|env|erase|erosion|escape|escaping|exception|exclud|expan|expos|extract|fault|feature|file|filesystem|forbid|framework|freeze|function|hash|heap|gemini|illegal|import|inconsisten|inject|injected|injection|inspect|interface|interleave|interpret|interrupt|kernel|key|keywords|launch|leak|let|link|lock|log|logical|loop|loss|lying|macbook|mark|match|memory|migrate|miss|modif| mv |new|nnpm|notif|null|oauth|object|operational|overhead|overloaded|override|overwrite|partial|pass|password|paste|persist|permission|perms|person|placeholder|plaintext|pnpm|policies|policy|privelege|process|produce|profile|prompt|proof|properties|property|protect|protocol|pure|race|raise|ram|recover|redact|regex|register|reject|relies|rely|remember|remind|reminder|remover|replace|repositories|repository|request|reset|respond|rest|revert|revers| rm |safety|salt|sanatize|schema|scheme|scope|search|secret|segfault|segment|select|sensitive|separat|sessionid|sets|shell|signal|snippet|spawned|spin|stack|stall|static|stream|subject|substitut|succeed|suite|supabase|symbol|system|tag|telemetry|temp|terminate|threshold|throttle|throws|time|timing|toggl|token|touch|trace|track|trail|unaware|undo|undocumented|unencrypt|uninstall|update|uri|url|usable|usememo|utilit|uuid|verify|violat|wait|waste|websearch|windsurf|wipe|xml")
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
