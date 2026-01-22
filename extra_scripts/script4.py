import csv
import pandas as pd

#Configuration
ISSUES_PATH = "./data/issues.csv"
CLUSTERS_PATH = "./data/cluster_assignments.csv"

#Load data
issues = pd.read_csv(ISSUES_PATH)
clusters = pd.read_csv(CLUSTERS_PATH)

issues["cluster"] = clusters["cluster"]
issues["severity_score"] = clusters["severity_score"]

#Inspect clusters
available_clusters = sorted(issues["cluster"].unique())
print("Available clusters:")
for cluster_id in available_clusters:
    count = (issues["cluster"] == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {count} issues")

cluster_choice = input("\nEnter the cluster ID to inspect (or 'all' for all clusters): ").strip()

if cluster_choice.lower() == "all":
    clusters_to_inspect = available_clusters
else:
    try:
        cluster_id = int(cluster_choice)
        if cluster_id not in available_clusters:
            print(f"Error: Cluster {cluster_id} not found.")
            exit(1)
        clusters_to_inspect = [cluster_id]
    except ValueError:
        print("Error: Please enter a valid cluster ID or 'all'.")
        exit(1)

for cluster_id in clusters_to_inspect:
    print("=" * 80)
    print(f"CLUSTER {cluster_id}")
    print("=" * 80)

    subset = issues[issues["cluster"] == cluster_id]
    # Sort by severity score (descending) so index 0 is most severe
    subset = subset.sort_values(by="severity_score", ascending=False).reset_index(drop=True)
    
    print(f"Total issues in cluster: {len(subset)}")
    
    severity_choice = input("\nEnter the severity ranking (1=most severe, or 'all' for all issues): ").strip()
    
    if severity_choice.lower() == "all":
        issues_to_show = subset
    else:
        try:
            severity_rank = int(severity_choice)
            if severity_rank < 1 or severity_rank > len(subset):
                print(f"Error: Severity ranking must be between 1 and {len(subset)}.")
                continue
            issues_to_show = subset.iloc[[severity_rank - 1]]
        except ValueError:
            print("Error: Please enter a valid severity ranking or 'all'.")
            continue

    for _, row in issues_to_show.iterrows():
        try:
            print(f"\n- {row['issue_title']}")
            print(f"  Severity Score: {row['severity_score']:.2f}")
            print(f"  {row['issue_body']}...")
        except Exception as e:
            print(f"Error processing row: {e}")
        pause = input("Press Enter to continue to the next issue...")
        