import sys
import csv
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

SEVERITY_THRESHOLD = 2.321120969699182
CLASSIFICATION_MAPPING = {
	0: "crash_failure",
	1: "hanging_freezing",
	2: "destruction_data_loss",
	3: "privacy_security",
	4: "auth_security",
}

WORD_REGEX = re.compile(r"accident[a-z]*|admin[a-z]*|apikey|audit[a-z]*|auth[a-z]*|breach[a-z]*|buffer[a-z]*|bypass[a-z]*|byte|chmod|clobber[a-z]*|complian[a-z]+|concern[a-z]*|concurren[a-z]+|condition|config[a-z]*|confirm[a-z]*|consent[a-z]*|console|constrain[a-z]*|consum[a-z]+|conver(?!sat)[a-z]+|corrupt[a-z]*|cpu|crash[a-z]*|credential|critical|\bdead\b|deadlock[a-z]*|decod[a-z]+|delet[a-z]+|denial|destr[a-z]+|disclos[a-z]+|disrupt[a-z]*|drop[a-z]*|dump[a-z]*|duplicat[a-z]+|edit(?!or)|eliminat[a-z]+|encod[a-z]+|encrypt[a-z]*|enforc[a-z]+|eras[a-z]+|escap[a-z]+|except[a-z]*|exclud[a-z]+|expan[a-z]+|expos[a-z]+|extract[a-z]*|fail[a-z]*|fault|forbid[a-z]*|freez[a-z]+|hash[a-z]*|heap|illegal[a-z]*|inconsisten[a-z]+|inject[a-z]*|interrupt[a-z]*|kernel|key|leak[a-z]*|lock[a-z]*|log[a-z]*|loop[a-z]*|loss|memory|modif[a-z]+|\bmv\b|null|oauth|overhead|overload[a-z]*|overrid[a-z]+|overwrit[a-z]+|password|persist|permission|perms|personal|plaintext|polic[a-z]+|privelege[a-z]*|protect[a-z]*|\brace|raise|\bram\b|recover[a-z]*|redact[a-z]*|remov[a-z]+|replac[a-z]+|reset|resource|revert|revers[a-z]+|\brm\b|safety|salt(?!y)|sanatiz[a-z]+|secret[a-z]*|segfault|sensitive|\bsets\b|sever[a-z]+|spawned|spin[a-z]*|stack|stall|undo|undocumented|unencrypt|uninstall|violat[a-z]+|wipe")

def tokenizer(text):
	return WORD_REGEX.findall(text.lower())

def main():
	if len(sys.argv) < 2:
		print("Usage: python classification.py <input_csv_file> [output_csv_file]")
		sys.exit(1)
	
	input_csv = sys.argv[1]
	output_csv = sys.argv[2] if len(sys.argv) > 2 else "./data/classified_issues.csv"
	
	# Load vectorizer
	with open("./data/tfidf_vectorizer.pkl", 'rb') as f:
		vectorizer = pickle.load(f)
	
	# Load cluster centroids
	centroids_df = pd.read_csv("./data/cluster_centroids.csv")
	centroids = centroids_df.values
	
	# Load issues from input CSV
	issues = []
	with open(input_csv, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			issues.append(row)
	
	# Classify issues
	results = []
	for issue in issues:
		title = (issue.get("issue_title", "") or "")
		body = (issue.get("issue_body", "") or "")
		text = title + " " + body
		
		# Compute TF-IDF vector
		tfidf_vector = vectorizer.transform([text]).toarray()[0]
		
		# Compute severity score
		severity_score = tfidf_vector.sum()
		
		# Check if above threshold
		if severity_score > SEVERITY_THRESHOLD:
			# Find closest centroid
			distances = cosine_distances([tfidf_vector], centroids)[0]
			closest_cluster = np.argmin(distances)
			classification = CLASSIFICATION_MAPPING.get(closest_cluster, "unknown")
			
			results.append({
				"issue_title": title,
				"issue_body": body,
				"severity_score": severity_score,
				"closest_cluster": closest_cluster,
				"classification": classification
			})
	
	# Save results
	results_df = pd.DataFrame(results)
	results_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
	
	print(f"Classified {len(results)} issues above threshold {SEVERITY_THRESHOLD}")
	print(f"Saved to {output_csv}")

if __name__ == "__main__":
	main()