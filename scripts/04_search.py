import os
import json
import time
from typing import List, Dict

import numpy as np
from qdrant_client import QdrantClient, models

# --- Library Imports with Error Handling ---
# We use try-except blocks so the script doesn't crash if you 
# haven't installed one of the optional libraries (like NMSLIB).
try: import faiss
except ImportError: faiss = None
try: import hnswlib
except ImportError: hnswlib = None
try: import nmslib
except ImportError: nmslib = None
try: import annoy
except ImportError: annoy = None

# --- Configuration Constants ---
RUNS_DIR = "runs"        # Where we store .run files for ir_measures
METRICS_DIR = "metrics"  # Where we store JSON timing data
EMB_DIR = "data/embeddings"
INDEX_DIR = "indexes"
TOPK = 100               # We retrieve the top 100 results per query

# Ensure directories exist
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------------------------------------
# WRAPPERS (The Adapter Pattern)
# Different libraries use different method names (e.g., .search vs .knn_query).
# These classes "wrap" the libraries so they all use a consistent .search() API.
# ---------------------------------------------------------

class QdrantWrapper:
    """
    Adapter for the Qdrant Vector Database.
    Unlike local libraries, this communicates over the network with Docker.
    """
    def __init__(self, collection_name, host="localhost", port=6333):
        # prefer_grpc=True is significantly faster for large data transfers (8.8M docs)
        self.client = QdrantClient(host=host, port=port, prefer_grpc=True)
        self.collection_name = collection_name

    def search(self, queries, k, ef=None):
        # Construct a list of search requests for Qdrant's batch API
        search_requests = [
            models.SearchRequest(
                vector=q.tolist(),
                limit=k,
                # efSearch (hnsw_ef) controls the speed/accuracy trade-off
                params=models.SearchParams(hnsw_ef=ef) if ef else None,
                with_payload=False # Only get IDs; we don't need text for the benchmark
            ) for q in queries
        ]
        
        # search_batch sends multiple queries in a single network request
        batch_results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=search_requests
        )
        
        # Format the output into NumPy arrays to match other libraries
        I_list, D_list = [], []
        for resp in batch_results:
            I_list.append([hit.id for hit in resp])
            D_list.append([hit.score for hit in resp])
        return np.array(D_list), np.array(I_list)

class HnswlibWrapper:
    """Adapter for the high-performance local hnswlib library."""
    def __init__(self, path, dim, num_elements):
        # 'ip' = Inner Product (equivalent to Cosine Similarity for normalized vectors)
        self.p = hnswlib.Index(space='ip', dim=dim)
        # We must specify max_elements to match the number of documents in the file
        self.p.load_index(path, max_elements=num_elements)
    
    def set_ef(self, ef):
        self.p.set_ef(ef)

    def search(self, queries, k):
        # returns (labels, distances)
        I, D = self.p.knn_query(queries, k=k)
        return D, I

class AnnoyWrapper:
    """Adapter for Spotify's Annoy (Approximate Nearest Neighbors Oh Yeah)."""
    def __init__(self, path, dim):
        # 'dot' = Dot Product similarity
        self.index = annoy.AnnoyIndex(dim, 'dot')
        self.index.load(path)
    
    def search(self, queries, k):
        I_list, D_list = [], []
        # Annoy does not support native batch search, so we loop over queries
        for q in queries:
            ids, dists = self.index.get_nns_by_vector(q, k, include_distances=True)
            I_list.append(ids)
            D_list.append(dists)
        return np.array(D_list), np.array(I_list)

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_json(path: str):
    with open(path, "r") as f: return json.load(f)

def save_json(path: str, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def trec_write(run_path, run_name, qids, doc_ids, I, D, is_qdrant=False):
    """
    Writes results in the standard TREC format used by Information Retrieval researchers.
    Format: [query_id] Q0 [doc_id] [rank] [score] [run_tag]
    """
    with open(run_path, "w") as f:
        for qi, qid in enumerate(qids):
            if qi >= len(I): break
            for rank in range(len(I[qi])):
                # If Qdrant, the point ID is already the document ID.
                # If FAISS/Annoy, 'I' contains an index we must map back to doc_ids.json.
                docid = str(I[qi, rank]) if is_qdrant else doc_ids[int(I[qi, rank])]
                score = float(D[qi, rank])
                # rank+1 because TREC uses 1-based ranking
                f.write(f"{qid} Q0 {docid} {rank+1} {score:.6f} {run_name}\n")

def latency_stats(latencies_ms: List[float]) -> Dict:
    """Calculates statistical performance metrics for the benchmark."""
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)), # Time 95% of queries took less than
        "qps": 1000.0 / float(arr.mean()) if arr.mean() > 0 else 0.0 # Queries Per Second
    }

def run_standard_search(index, q, topk, batch_size=1):
    """
    Executes the search and measures precisely how many milliseconds it takes.
    batch_size=1 is used for 'real-time' latency testing.
    batch_size=64+ is used for 'throughput' testing.
    """
    I_all, D_all, lat_ms = [], [], []
    for i in range(0, q.shape[0], batch_size):
        qi = q[i:i+batch_size]
        
        # High-resolution timer
        t0 = time.perf_counter()
        D, I = index.search(qi, topk)
        t1 = time.perf_counter()
        
        # Calculate time per individual query in the batch
        batch_time = (t1 - t0) * 1000.0
        for _ in range(len(qi)):
            lat_ms.append(batch_time / len(qi))
            
        I_all.append(I)
        D_all.append(D)
    return np.vstack(I_all), np.vstack(D_all), lat_ms

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    # 1. Load Pre-computed Embeddings and Metadata
    query_emb = np.load(os.path.join(EMB_DIR, "query_emb.npy")).astype(np.float32)
    query_ids = load_json(os.path.join(EMB_DIR, "query_ids.json"))
    doc_ids = load_json(os.path.join(EMB_DIR, "doc_ids.json"))
    num_docs = len(doc_ids)
    DIM = query_emb.shape[1]

    print(f"Loaded {len(query_ids)} queries. Dim={DIM}. Collection Size={num_docs}")

    # ==========================
    # TEST 1: QDRANT (The "Nearly" Challenger)
    # ==========================
    print("\n--- QDRANT ---")
    try:
        q_wrap = QdrantWrapper(collection_name="nearly_bench")
        # We test two EF values: 32 (Fast/Inaccurate) and 128 (Slow/Accurate)
        for ef in [32, 128]: 
            name = f"qdrant_ef{ef}"
            print(f"Running {name}...")
            # We use a lambda to pass the 'ef' parameter through the standard search helper
            I, D, lat = run_standard_search(
                type('Obj', (object,), {'search': lambda q, k: q_wrap.search(q, k, ef=ef)}), 
                query_emb, TOPK, batch_size=64
            )
            trec_write(os.path.join(RUNS_DIR, name+".run"), name, query_ids, None, I, D, is_qdrant=True)
            save_json(os.path.join(METRICS_DIR, name+".json"), latency_stats(lat))
    except Exception as e:
        print(f"Skipping Qdrant: {e}")

    # ==========================
    # TEST 2: FAISS HNSW
    # ==========================
    path = os.path.join(INDEX_DIR, "faiss_hnsw_ip.index")
    if os.path.exists(path) and faiss:
        print("\n--- FAISS HNSW ---")
        index = faiss.read_index(path)
        for ef in [64, 128]:
            index.hnsw.efSearch = ef
            name = f"faiss_hnsw_ef{ef}"
            print(f"Running {name}...")
            I, D, lat = run_standard_search(index, query_emb, TOPK, batch_size=100)
            trec_write(os.path.join(RUNS_DIR, name+".run"), name, query_ids, doc_ids, I, D)
            save_json(os.path.join(METRICS_DIR, name+".json"), latency_stats(lat))

    # ==========================
    # TEST 3: ANNOY
    # ==========================
    path = os.path.join(INDEX_DIR, "annoy_ip.ann")
    if os.path.exists(path) and annoy:
        print("\n--- ANNOY ---")
        index = AnnoyWrapper(path, DIM)
        name = "annoy"
        print(f"Running {name}...")
        I, D, lat = run_standard_search(index, query_emb, TOPK)
        trec_write(os.path.join(RUNS_DIR, name+".run"), name, query_ids, doc_ids, I, D)
        save_json(os.path.join(METRICS_DIR, name+".json"), latency_stats(lat))

    print(f"\n Search phase complete. Next: 'python scripts/05_evaluate.py'.")

if __name__ == "__main__":
    main()