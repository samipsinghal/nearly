"""
01_prepare_data.py

Loads MS MARCO passage dataset using ir_datasets,
selects a subset of documents, and saves:
- documents
- queries
- qrels

Output is written to data/raw/ as JSON files.
"""

import json
import os
from itertools import islice

import ir_datasets
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = "data/raw"
DATASET_NAME = "msmarco-passage/dev"
N_DOCS = 500_000        # change to 500_000 if time allows
N_QUERIES = 7000        # dev query subset size (set None to use all available queries)

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

docs_out = os.path.join(OUTPUT_DIR, "documents.json")
queries_out = os.path.join(OUTPUT_DIR, "queries.json")
qrels_out = os.path.join(OUTPUT_DIR, "qrels.json")

print(f"Loading dataset: {DATASET_NAME}")
dataset = ir_datasets.load(DATASET_NAME)

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
print(f"Loading documents (max {N_DOCS})...")
documents = []

for doc in tqdm(islice(dataset.docs_iter(), N_DOCS)):
    documents.append({
        "doc_id": doc.doc_id,
        "text": doc.text
    })

print(f"Loaded {len(documents)} documents")

with open(docs_out, "w") as f:
    json.dump(documents, f)

# -----------------------------
# LOAD QUERIES
# -----------------------------
print("Loading queries...")
queries = []

query_iter = dataset.queries_iter()
if N_QUERIES is not None:
    query_iter = islice(query_iter, N_QUERIES)

for q in tqdm(query_iter):
    queries.append({
        "query_id": q.query_id,
        "text": q.text
    })

print(f"Loaded {len(queries)} queries")

with open(queries_out, "w") as f:
    json.dump(queries, f)

# -----------------------------
# LOAD QRELS
# -----------------------------
print("Loading qrels...")
qrels = []

for r in tqdm(dataset.qrels_iter()):
    qrels.append({
        "query_id": r.query_id,
        "doc_id": r.doc_id,
        "relevance": r.relevance
    })

print(f"Loaded {len(qrels)} qrels")

with open(qrels_out, "w") as f:
    json.dump(qrels, f)

print("Done. Data written to data/raw/")