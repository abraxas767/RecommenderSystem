# Project Name

This project implements a content-based recommendation system using the TF-IDF method. It processes a ZIM file, calculates similarities between text documents, and allows you to find similar documents for a given document.

## Prerequisites

Ensure the following prerequisites are met:

- Python >=3.10.10 installed
- Required Python packages: (see `requirements.txt`)
- An `output.zim` file must be present in the `corpus/` directory.

## Usage

### 1. Preprocessing the Corpus

The `preprocessing.py` script processes the `output.zim` file and stores the processed data in a SQLite database in the `corpus/` directory as `corpus.db`.

```bash
python preprocessing.py
```

### 2. Calculating Similarities
The calculate_similarities.py script creates a TF-IDF matrix based on the preprocessed data in corpus.db and calculates the similarities between the documents.
```bash
python calculate_similarities.py
```
### 3. Retrieving Recommendations
With the recommend.py script, you can retrieve the top 5 recommendations for similar documents based on a given document ID.
```bash
python recommend.py
```
All titles can be looked up at https://www.gutenberg.org/
