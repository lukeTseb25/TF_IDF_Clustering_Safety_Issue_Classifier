import pandas as pd
import numpy as np

# Read the vectors and issues
vectors_df = pd.read_csv('data/tfidf_vectors.csv')
issues_df = pd.read_csv('data/issues.csv')

vectors_array = np.array(vectors_df.values)
scores = vectors_array.sum(axis=1)

# Sort scores in descending order to get rankings
sorted_indices = np.argsort(scores)[::-1]

# Get user input for which rank they want
print(f"Total number of issues: {len(scores)}")
while True:
    try:
        n = int(input(f"Enter a number between 1 and {len(scores)}: "))
        if 1 <= n <= len(scores):
            break
        else:
            print(f"Please enter a number between 1 and {len(scores)}")
    except ValueError:
        print("Please enter a valid integer")

# Get the index of the nth highest distance
nth_idx = sorted_indices[n - 1]

# Get the corresponding issue
issue = issues_df.iloc[nth_idx]

# Display the results
print(f"\nVector with {n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'} highest saftey concern score:")
print(f"Score: {scores[nth_idx]:.6f}")
print(f"Row Index: {nth_idx}")
print(f"\nIssue Title: {issue['issue_title']}")
print(f"\nIssue Body:")
print(issue['issue_body'])
