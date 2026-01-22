# Issue Classification System

This project provides an automated system for classifying software issues into five security and reliability categories using TF-IDF vectorization and K-Means clustering.

## Issue Categories

- **crash_failure**: System crashes and application failures
- **hanging_freezing**: Performance degradation and resource deadlocks
- **destruction_data_loss**: Data corruption and permanent data loss
- **privacy_security**: Privacy violations and data exposure
- **auth_security**: Authentication and authorization vulnerabilities

## Quick Start (Using Pre-trained Model)

If you want to classify issues without retraining the model, simply run:

```bash
python .\classification.py .\data\issues.csv
```

This will classify the issues in your CSV file using the existing `cluster_centroids.csv` file and output results to `classified_issues.csv`.

## Full Workflow (Including Model Training)

### Step 1: Prepare Your Data

Place your CSV file containing issues into the `./data/` directory. The file must contain the following columns:

- `issue_title`: The title of the issue
- `issue_body`: The detailed description of the issue

### Step 2: Generate TF-IDF Vectors

Extract security-related keywords and generate TF-IDF vectors from your issues:

```bash
python .\vocab_extraction.py
```

This script:
- Reads `./data/issues.csv`
- Extracts security and reliability-related keywords
- Computes TF-IDF vectors
- Outputs `./data/tfidf_vectors.csv`
- Generates an elbow plot to `./data/idf_elbow.png`

### Step 3: Generate Cluster Centroids

Generate K-Means cluster centroids from the TF-IDF vectors:

```bash
python .\kmeans_centroid_generation.py
```

This script:
- Reads `./data/tfidf_vectors.csv` and `./data/issues.csv`
- Computes severity scores for each issue
- Runs K-Means clustering (default k=5)
- Outputs cluster assignments to `./data/cluster_assignments.csv`
- Outputs cluster centroids to `./data/cluster_centroids.csv`
- Generates elbow plots for both K-Means and severity analysis

### Step 4: Create IDF Vectorizer

Build and serialize the TF-IDF vectorizer for future classification:

```bash
python .\create_idf_vectorizer.py
```

This script:
- Reads `./data/issues.csv`
- Creates a TF-IDF vectorizer with the same keyword extraction rules
- Saves the vectorizer to `./data/tfidf_vectorizer.pkl`
- Exports IDF values to `./data/idf_values.csv`

### Step 5: Classify Issues

Classify new or existing issues:

```bash
python .\classification.py .\data\issues.csv
```

You can optionally specify an output filename:

```bash
python .\classification.py .\data\issues.csv output_filename.csv
```

This script:
- Reads issues from the input CSV
- Vectorizes them using the saved `tfidf_vectorizer.pkl`
- Classifies them using `cluster_centroids.csv`
- Outputs results to `classified_issues.csv` (or your specified filename)

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Package Details

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computations
- **scikit-learn**: TF-IDF vectorization, K-Means clustering, and cosine distance metrics
- **matplotlib**: Visualization of clustering results and elbow plots

## Project Structure

```
.
├── classification.py                 # Classify issues using pre-trained model
├── create_idf_vectorizer.py         # Generate and serialize TF-IDF vectorizer
├── kmeans_centroid_generation.py    # Generate cluster centroids
├── vocab_extraction.py              # Extract keywords and generate TF-IDF vectors
├── data/
│   ├── issues.csv                   # Input: raw issues (required)
│   ├── tfidf_vectors.csv           # Intermediate: TF-IDF vectors
│   ├── tfidf_vectorizer.pkl        # Generated: serialized vectorizer
│   ├── idf_values.csv              # Generated: IDF values
│   ├── cluster_centroids.csv       # Generated: cluster center vectors
│   ├── cluster_assignments.csv     # Generated: issue-to-cluster mappings
│   └── cluster_histograms/         # Generated: visualization plots
└── README.md                        # This file
```

## Notes

- The system uses a domain-specific keyword regex to extract security and reliability-related terms from issue text
- Cluster centroids are automatically determined using the elbow method
- Classification is performed using cosine distance metrics
- A severity threshold (2.321) is used to distinguish between critical and non-critical issues
- The `extra_scripts/` directory contains previous and alternative versions of scripts developed during this project, as well as utility scripts for data evaluation and understanding
