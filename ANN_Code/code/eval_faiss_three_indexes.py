#!/usr/bin/env python3
"""
I evaluate downstream retrieval quality for (1) FlatIP oracle, (2) IVF-Flat, (3) IVF-PQ
on MS MARCO dev queries.

Outputs:
- results_eval.csv : sweep results with recall + ranking metrics + latency
- oracle_topk.npz  : cached oracle neighbors from FlatIP (so I don't recompute every run)
"""

import os
import time
import json
import numpy as np
import faiss
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------
# CONFIG (edit these paths)
# ----------------------------
INDEX_DIR = os.path.expanduser("~/data/indexes_full")

FLAT_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_flat_ip.index")
IVF_FLAT_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_ivf_flat_ip_nlist16384.index")
IVF_PQ_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_ivf_pq_ip_nlist16384_m64_b8.index")

DATA_DIR = os.path.expanduser("~/data")
MSMARCO_DIR = os.path.join(DATA_DIR, "msmarco")

QUERY_EMB_PATH = os.path.join(DATA_DIR, "embeddings", "query_emb_dev.npy")
DEV_QIDS_PATH = os.path.join(MSMARCO_DIR, "dev_query_ids.txt")
QRELS_PATH = os.path.join(MSMARCO_DIR, "qrels.dev.tsv")

# Important: mapping FAISS internal ids -> external doc ids used in qrels
DOCIDS_PATH = os.path.join(MSMARCO_DIR, "doc_ids.txt")

CACHE_DIR = os.path.join(INDEX_DIR, "eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ORACLE_CACHE_PATH = os.path.join(CACHE_DIR, "oracle_topk.npz")

OUT_CSV = os.path.join(CACHE_DIR, "results_eval.csv")

# Evaluation cutoffs (edit as needed)
CANDIDATE_K = 200        # retrieve topK candidates from each index
ORACLE_K = 200           # oracle neighbors for recall@K comparison
NDCG_K = 10
MRR_K = 10
MAP_K = 100

# Sweeps (reasonable starting set)
IVF_NPROBE_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
PQ_NPROBE_SWEEP  = [1, 2, 4, 8, 16, 32, 64, 128]

# Batch sizes: trade memory vs speed
QUERY_BATCH = 256

# ----------------------------
# Helpers: qrels + metrics
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
    """
    I read MS MARCO qrels: qid, 0, docid, rel
    Returns: qrels[qid][docid] = rel (int)
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            rel = int(rel)
            qrels.setdefault(qid, {})[docid] = rel
    return qrels

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0**rels - 1.0) * discounts))

def ndcg_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    rels = np.array([qrel.get(d, 0) for d in ranked_docids[:k]], dtype=np.float32)
    dcg = dcg_at_k(rels, k)
    # ideal
    ideal_rels = np.array(sorted(qrel.values(), reverse=True), dtype=np.float32)
    idcg = dcg_at_k(ideal_rels, k)
    return 0.0 if idcg == 0 else float(dcg / idcg)

def mrr_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked_docids[:k], start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def ap_at_k(ranked_docids: List[str], qrel: Dict[str, int], k: int) -> float:
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_docids[:k], start=1):
        if qrel.get(d, 0) > 0:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / hits

def recall_at_k(approx_ids: np.ndarray, oracle_ids: np.ndarray, k: int) -> float:
    """
    I compute geometric recall@k: fraction of oracle top-k present in approx top-k.
    Inputs are int64 FAISS ids.
    """
    a = set(map(int, approx_ids[:k]))
    o = set(map(int, oracle_ids[:k]))
    return 0.0 if len(o) == 0 else (len(a.intersection(o)) / len(o))

def percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, p))

# ----------------------------
# FAISS search wrappers
# ----------------------------

@dataclass
class SearchResult:
    ids: np.ndarray      # [nq, k] int64
    scores: np.ndarray   # [nq, k] float32
    per_query_ms: np.ndarray  # [nq] float32

def batched_search(index, queries: np.ndarray, k: int) -> SearchResult:
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    scores = np.empty((nq, k), dtype=np.float32)
    per_query_ms = np.empty((nq,), dtype=np.float32)

    for start in range(0, nq, QUERY_BATCH):
        end = min(start + QUERY_BATCH, nq)
        q = queries[start:end]
        t0 = time.perf_counter()
        s, i = index.search(q, k)  # FAISS returns (scores, ids)
        t1 = time.perf_counter()

        scores[start:end] = s.astype(np.float32, copy=False)
        ids[start:end] = i.astype(np.int64, copy=False)

        # I assign the batch latency equally per query (good enough for aggregated p50/p95)
        batch_ms = (t1 - t0) * 1000.0
        per_query_ms[start:end] = batch_ms / (end - start)

    return SearchResult(ids=ids, scores=scores, per_query_ms=per_query_ms)

def load_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing index: {path}")
    return faiss.read_index(path)

# ----------------------------
# Main evaluation
# ----------------------------

def main():
    # I load data
    if not os.path.exists(QUERY_EMB_PATH):
        raise FileNotFoundError(f"Missing query embeddings: {QUERY_EMB_PATH}")

    queries = np.load(QUERY_EMB_PATH, mmap_mode="r")
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)

    qids = load_qids(DEV_QIDS_PATH)
    docids = load_docids(DOCIDS_PATH)
    qrels = load_qrels(QRELS_PATH)

    assert queries.shape[0] == qids.shape[0], "Query embeddings and qid list must align"
    print(f"I loaded {queries.shape[0]:,} queries, dim={queries.shape[1]}, dtype={queries.dtype}")
    print(f"I loaded {len(docids):,} docids (must equal corpus vectors)")
    print(f"I loaded qrels for {len(qrels):,} queries")

    # I load indexes
    flat = load_index(FLAT_INDEX_PATH)
    ivf = load_index(IVF_FLAT_INDEX_PATH)
    pq  = load_index(IVF_PQ_INDEX_PATH)

    # Threading
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())

    # Oracle cache (so I don't recompute Flat every time)
    if os.path.exists(ORACLE_CACHE_PATH):
        print(f"I found oracle cache: {ORACLE_CACHE_PATH}")
        cached = np.load(ORACLE_CACHE_PATH)
        oracle_ids = cached["oracle_ids"]
        oracle_scores = cached["oracle_scores"]
    else:
        print("I am computing oracle (FlatIP) topK once and caching it...")
        oracle = batched_search(flat, queries, ORACLE_K)
        oracle_ids = oracle.ids
        oracle_scores = oracle.scores
        np.savez_compressed(ORACLE_CACHE_PATH, oracle_ids=oracle_ids, oracle_scores=oracle_scores)
        print(f"I wrote oracle cache: {ORACLE_CACHE_PATH}")

    # CSV header
    header = [
        "method", "param_name", "param_value",
        f"recall@{ORACLE_K}", f"ndcg@{NDCG_K}", f"mrr@{MRR_K}", f"map@{MAP_K}",
        "lat_p50_ms", "lat_p95_ms", "qps"
    ]

    rows: List[List] = []
    print("I am starting sweeps...")

    # ---------- IVF-Flat sweep ----------
    if hasattr(ivf, "nprobe"):
        for nprobe in IVF_NPROBE_SWEEP:
            ivf.nprobe = int(nprobe)
            print(f"\n[IVF-Flat] nprobe={nprobe}")
            res = batched_search(ivf, queries, CANDIDATE_K)

            # Metrics aggregation
            recalls = []
            ndcgs = []
            mrrs = []
            aps = []

            for qi in range(queries.shape[0]):
                qid = str(qids[qi])
                qrel = qrels.get(qid, {})
                approx = res.ids[qi]
                oracle = oracle_ids[qi]

                recalls.append(recall_at_k(approx, oracle, ORACLE_K))

                # convert internal ids -> external docids for ranking metrics
                ranked_docids = [str(docids[int(d)]) for d in approx if int(d) >= 0]
                ndcgs.append(ndcg_at_k(ranked_docids, qrel, NDCG_K))
                mrrs.append(mrr_at_k(ranked_docids, qrel, MRR_K))
                aps.append(ap_at_k(ranked_docids, qrel, MAP_K))

            recalls = np.array(recalls, dtype=np.float32)
            ndcgs = np.array(ndcgs, dtype=np.float32)
            mrrs = np.array(mrrs, dtype=np.float32)
            aps = np.array(aps, dtype=np.float32)

            lat_p50 = percentile(res.per_query_ms, 50)
            lat_p95 = percentile(res.per_query_ms, 95)
            qps = 1000.0 / np.mean(res.per_query_ms)

            rows.append([
                "ivf_flat", "nprobe", nprobe,
                float(recalls.mean()), float(ndcgs.mean()), float(mrrs.mean()), float(aps.mean()),
                lat_p50, lat_p95, qps
            ])

            print(f"  recall@{ORACLE_K}: {recalls.mean():.4f}")
            print(f"  nDCG@{NDCG_K}   : {ndcgs.mean():.4f}")
            print(f"  MRR@{MRR_K}     : {mrrs.mean():.4f}")
            print(f"  MAP@{MAP_K}     : {aps.mean():.4f}")
            print(f"  latency p50/p95 : {lat_p50:.3f} / {lat_p95:.3f} ms")
            print(f"  QPS (approx)    : {qps:.1f}")

    # ---------- IVF-PQ sweep ----------
    if hasattr(pq, "nprobe"):
        for nprobe in PQ_NPROBE_SWEEP:
            pq.nprobe = int(nprobe)
            print(f"\n[IVF-PQ] nprobe={nprobe}")
            res = batched_search(pq, queries, CANDIDATE_K)

            recalls = []
            ndcgs = []
            mrrs = []
            aps = []

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

            recalls = np.array(recalls, dtype=np.float32)
            ndcgs = np.array(ndcgs, dtype=np.float32)
            mrrs = np.array(mrrs, dtype=np.float32)
            aps = np.array(aps, dtype=np.float32)

            lat_p50 = percentile(res.per_query_ms, 50)
            lat_p95 = percentile(res.per_query_ms, 95)
            qps = 1000.0 / np.mean(res.per_query_ms)

            rows.append([
                "ivf_pq", "nprobe", nprobe,
                float(recalls.mean()), float(ndcgs.mean()), float(mrrs.mean()), float(aps.mean()),
                lat_p50, lat_p95, qps
            ])

            print(f"  recall@{ORACLE_K}: {recalls.mean():.4f}")
            print(f"  nDCG@{NDCG_K}   : {ndcgs.mean():.4f}")
            print(f"  MRR@{MRR_K}     : {mrrs.mean():.4f}")
            print(f"  MAP@{MAP_K}     : {aps.mean():.4f}")
            print(f"  latency p50/p95 : {lat_p50:.3f} / {lat_p95:.3f} ms")
            print(f"  QPS (approx)    : {qps:.1f}")

    # Write CSV
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print(f"\nI wrote results to: {OUT_CSV}")
    print(f"I wrote oracle cache to: {ORACLE_CACHE_PATH}")
    print("Done.")

if __name__ == "__main__":
    main()

