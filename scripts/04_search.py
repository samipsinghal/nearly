"""
04_search.py

Runs retrieval on FAISS indexes and writes:
- TREC run files (runs/*.run)
- latency summaries (metrics/*.json)

Inputs:
- data/embeddings/query_emb.npy
- data/embeddings/query_ids.json
- data/embeddings/doc_ids.json
- indexes/faiss_hnsw_ip.index
- indexes/faiss_ivf_flat_ip.index
- indexes/faiss_ivf_pq_ip.index

Notes:
- Embeddings are L2-normalized => IP behaves like cosine similarity.
"""

import os
import json
import time
from typing import List, Dict

import numpy as np

RUNS_DIR = "runs"
METRICS_DIR = "metrics"
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

EMB_DIR = "data/embeddings"
INDEX_DIR = "indexes"

TOPK = 200
SEED = 42

HNSW_EF_SEARCH = [32, 64, 128, 256]
IVF_NPROBE = [1, 4, 8, 16, 32, 64]

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def trec_write(run_path: str, run_name: str, qids: List[str], doc_ids: List[str], I: np.ndarray, D: np.ndarray):
    """
    Writes TREC format:
    qid Q0 docid rank score runname
    """
    with open(run_path, "w") as f:
        for qi, qid in enumerate(qids):
            for rank in range(I.shape[1]):
                di = int(I[qi, rank])
                if di < 0:
                    continue
                docid = doc_ids[di]
                score = float(D[qi, rank])
                f.write(f"{qid} Q0 {docid} {rank+1} {score:.6f} {run_name}\n")

def latency_stats(latencies_ms: List[float]) -> Dict:
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }

def search_index(index, q: np.ndarray, topk: int) -> (np.ndarray, np.ndarray, List[float]):
    """
    Returns (I, D, per_query_latency_ms)
    """
    I_all = []
    D_all = []
    lat_ms = []
    # Query-by-query timing (simple and robust)
    for i in range(q.shape[0]):
        qi = q[i:i+1]
        t0 = time.perf_counter()
        D, I = index.search(qi, topk)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
        I_all.append(I)
        D_all.append(D)
    I = np.vstack(I_all)
    D = np.vstack(D_all)
    return I, D, lat_ms

def main():
    import faiss

    query_emb = np.load(os.path.join(EMB_DIR, "query_emb.npy")).astype(np.float32)
    query_ids = load_json(os.path.join(EMB_DIR, "query_ids.json"))
    doc_ids = load_json(os.path.join(EMB_DIR, "doc_ids.json"))

    print("query_emb:", query_emb.shape, query_emb.dtype)
    print("num queries:", len(query_ids), "num docs:", len(doc_ids))

    # -------------------------
    # HNSW
    # -------------------------
    hnsw_path = os.path.join(INDEX_DIR, "faiss_hnsw_ip.index")
    if os.path.exists(hnsw_path):
        index = faiss.read_index(hnsw_path)
        for ef in HNSW_EF_SEARCH:
            # set efSearch
            index.hnsw.efSearch = ef
            run_name = f"faiss_hnsw_ip_ef{ef}_k{TOPK}"
            run_path = os.path.join(RUNS_DIR, run_name + ".run")
            metrics_path = os.path.join(METRICS_DIR, run_name + "_latency.json")

            print(f"\nSearching HNSW efSearch={ef} ...")
            I, D, lat = search_index(index, query_emb, TOPK)
            trec_write(run_path, run_name, query_ids, doc_ids, I, D)

            stats = latency_stats(lat)
            stats.update({"method": "faiss_hnsw_ip", "efSearch": ef, "topk": TOPK})
            save_json(metrics_path, stats)
            print("latency:", stats)
            print("run file:", run_path)
    else:
        print("HNSW index not found:", hnsw_path)

    # -------------------------
    # IVF-Flat
    # -------------------------
    ivf_flat_path = os.path.join(INDEX_DIR, "faiss_ivf_flat_ip.index")
    if os.path.exists(ivf_flat_path):
        index = faiss.read_index(ivf_flat_path)
        for nprobe in IVF_NPROBE:
            index.nprobe = nprobe
            run_name = f"faiss_ivf_flat_ip_nprobe{nprobe}_k{TOPK}"
            run_path = os.path.join(RUNS_DIR, run_name + ".run")
            metrics_path = os.path.join(METRICS_DIR, run_name + "_latency.json")

            print(f"\nSearching IVF-Flat nprobe={nprobe} ...")
            I, D, lat = search_index(index, query_emb, TOPK)
            trec_write(run_path, run_name, query_ids, doc_ids, I, D)

            stats = latency_stats(lat)
            stats.update({"method": "faiss_ivf_flat_ip", "nprobe": nprobe, "topk": TOPK})
            save_json(metrics_path, stats)
            print("latency:", stats)
            print("run file:", run_path)
    else:
        print("IVF-Flat index not found:", ivf_flat_path)

    # -------------------------
    # IVF-PQ
    # -------------------------
    ivf_pq_path = os.path.join(INDEX_DIR, "faiss_ivf_pq_ip.index")
    if os.path.exists(ivf_pq_path):
        index = faiss.read_index(ivf_pq_path)
        for nprobe in IVF_NPROBE:
            index.nprobe = nprobe
            run_name = f"faiss_ivf_pq_ip_nprobe{nprobe}_k{TOPK}"
            run_path = os.path.join(RUNS_DIR, run_name + ".run")
            metrics_path = os.path.join(METRICS_DIR, run_name + "_latency.json")

            print(f"\nSearching IVF-PQ nprobe={nprobe} ...")
            I, D, lat = search_index(index, query_emb, TOPK)
            trec_write(run_path, run_name, query_ids, doc_ids, I, D)

            stats = latency_stats(lat)
            stats.update({"method": "faiss_ivf_pq_ip", "nprobe": nprobe, "topk": TOPK})
            save_json(metrics_path, stats)
            print("latency:", stats)
            print("run file:", run_path)
    else:
        print("IVF-PQ index not found:", ivf_pq_path)

    print("\nDone. Runs in runs/ and latency in metrics/")

if __name__ == "__main__":
    main()
