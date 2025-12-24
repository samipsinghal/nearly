#!/usr/bin/env python3
"""
I evaluate downstream retrieval quality for HNSW on MS MARCO dev queries.
"""

import os
import time
import numpy as np
import faiss
from dataclasses import dataclass
from typing import Dict, List

# ----------------------------
# CONFIG (Updated for HNSW)
# ----------------------------
INDEX_DIR = os.path.expanduser("~/data/indexes_full")

# PATH TO YOUR NEW INDEX
HNSW_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_hnsw_ip_M16_efC100.index")
FLAT_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_flat_ip.index") # Kept for Oracle calculation

DATA_DIR = os.path.expanduser("~/data")
MSMARCO_DIR = os.path.join(DATA_DIR, "msmarco")

QUERY_EMB_PATH = os.path.join(DATA_DIR, "embeddings", "query_emb_dev.npy")
DEV_QIDS_PATH = os.path.join(MSMARCO_DIR, "dev_query_ids.txt")
QRELS_PATH = os.path.join(MSMARCO_DIR, "qrels.dev.tsv")
DOCIDS_PATH = os.path.join(MSMARCO_DIR, "doc_ids.txt")

CACHE_DIR = os.path.join(INDEX_DIR, "eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ORACLE_CACHE_PATH = os.path.join(CACHE_DIR, "oracle_topk.npz")
OUT_CSV = os.path.join(CACHE_DIR, "results_hnsw_eval.csv")

# Evaluation cutoffs
CANDIDATE_K = 200        
ORACLE_K = 200           
NDCG_K = 10
MRR_K = 10
MAP_K = 100

# HNSW Sweep: efSearch controls the search depth (accuracy vs speed)
EF_SEARCH_SWEEP = [16, 32, 64, 128, 256, 512]

QUERY_BATCH = 256

# ----------------------------
# Helpers (Same as before)
# ----------------------------

def load_qids(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        qids = [line.strip() for line in f if line.strip()]
    return np.array(qids, dtype=object)

def load_docids(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        docids = [line.strip() for line in f if line.strip()]
    return np.array(docids, dtype=object)

def load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4: continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0**rels - 1.0) * discounts))

def ndcg_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    rels = np.array([qrel.get(d, 0) for d in ranked_docids[:k]], dtype=np.float32)
    dcg = dcg_at_k(rels, k)
    ideal_rels = np.array(sorted(qrel.values(), reverse=True), dtype=np.float32)
    idcg = dcg_at_k(ideal_rels, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)

def mrr_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked_docids[:k], start=1):
        if qrel.get(d, 0) > 0: return 1.0 / i
    return 0.0

def ap_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_docids[:k], start=1):
        if qrel.get(d, 0) > 0:
            hits += 1
            sum_prec += hits / i
    return 0.0 if hits == 0 else sum_prec / hits

def recall_at_k(approx_ids: np.ndarray, oracle_ids: np.ndarray, k: int) -> float:
    a = set(map(int, approx_ids[:k]))
    o = set(map(int, oracle_ids[:k]))
    return 0.0 if len(o) == 0 else (len(a.intersection(o)) / len(o))

def percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0: return 0.0
    return float(np.percentile(x, p))

# ----------------------------
# FAISS search wrappers
# ----------------------------

@dataclass
class SearchResult:
    ids: np.ndarray      
    scores: np.ndarray   
    per_query_ms: np.ndarray 

def batched_search(index, queries: np.ndarray, k: int) -> SearchResult:
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    scores = np.empty((nq, k), dtype=np.float32)
    per_query_ms = np.empty((nq,), dtype=np.float32)

    for start in range(0, nq, QUERY_BATCH):
        end = min(start + QUERY_BATCH, nq)
        q = queries[start:end]
        t0 = time.perf_counter()
        s, i = index.search(q, k) 
        t1 = time.perf_counter()

        scores[start:end] = s.astype(np.float32, copy=False)
        ids[start:end] = i.astype(np.int64, copy=False)

        batch_ms = (t1 - t0) * 1000.0
        per_query_ms[start:end] = batch_ms / (end - start)

    return SearchResult(ids=ids, scores=scores, per_query_ms=per_query_ms)

def load_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index: {path}")
    print(f"Loading index from {path}...")
    return faiss.read_index(path)

# ----------------------------
# Main evaluation
# ----------------------------

def main():
    # Load Data
    queries = np.load(QUERY_EMB_PATH, mmap_mode="r").astype(np.float32)
    qids = load_qids(DEV_QIDS_PATH)
    docids = load_docids(DOCIDS_PATH)
    qrels = load_qrels(QRELS_PATH)

    print(f"Loaded {queries.shape[0]:,} queries.")

    # Load Indexes
    hnsw = load_index(HNSW_INDEX_PATH)
    flat = load_index(FLAT_INDEX_PATH)

    # Threading
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())

    # Oracle Cache
    if os.path.exists(ORACLE_CACHE_PATH):
        print(f"Found oracle cache: {ORACLE_CACHE_PATH}")
        cached = np.load(ORACLE_CACHE_PATH)
        oracle_ids = cached["oracle_ids"]
    else:
        print("Computing oracle (FlatIP) topK...")
        oracle = batched_search(flat, queries, ORACLE_K)
        oracle_ids = oracle.ids
        np.savez_compressed(ORACLE_CACHE_PATH, oracle_ids=oracle_ids, oracle_scores=oracle.scores)

    # CSV Header
    header = [
        "method", "param_name", "param_value",
        f"recall@{ORACLE_K}", f"ndcg@{NDCG_K}", f"mrr@{MRR_K}", f"map@{MAP_K}",
        "lat_p50_ms", "lat_p95_ms", "qps"
    ]
    rows = []

    print("Starting HNSW efSearch sweep...")

    # ---------- HNSW Sweep ----------
    for ef in EF_SEARCH_SWEEP:
        # KEY CHANGE: Set efSearch parameter
        hnsw.hnsw.efSearch = int(ef)
        
        print(f"\n[HNSW] efSearch={ef}")
        res = batched_search(hnsw, queries, CANDIDATE_K)

        recalls, ndcgs, mrrs, aps = [], [], [], []

        for qi in range(queries.shape[0]):
            qid = str(qids[qi])
            qrel = qrels.get(qid, {})
            approx = res.ids[qi]
            oracle = oracle_ids[qi]

            recalls.append(recall_at_k(approx, oracle, ORACLE_K))

            ranked_docids = [str(docids[int(d)]) for d in approx if int(d) >= 0]
            ndcgs.append(ndcg_at_k(ranked_docids, qrel, NDCG_K))
            mrrs.append(mrr_at_k(ranked_docids, qrel, MRR_K))
            aps.append(ap_at_k(ranked_docids, qrel, MAP_K))

        lat_p50 = percentile(res.per_query_ms, 50)
        lat_p95 = percentile(res.per_query_ms, 95)
        qps = 1000.0 / np.mean(res.per_query_ms)
        
        mean_recall = np.mean(recalls)
        mean_ndcg = np.mean(ndcgs)
        mean_mrr = np.mean(mrrs)
        mean_ap = np.mean(aps)

        rows.append([
            "hnsw", "efSearch", ef,
            mean_recall, mean_ndcg, mean_mrr, mean_ap,
            lat_p50, lat_p95, qps
        ])

        print(f"  recall@{ORACLE_K}: {mean_recall:.4f}")
        print(f"  nDCG@{NDCG_K}    : {mean_ndcg:.4f}")
        print(f"  QPS             : {qps:.1f}")

    # Write CSV
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"\nSaved HNSW results to: {OUT_CSV}")

if __name__ == "__main__":
    main()
