#!/usr/bin/env python3
"""
Evaluate IVF-HNSW ANN index on MS MARCO dev queries.

Metrics:
- recall@200 (geometric, vs FlatIP oracle)
- nDCG@10
- MRR@10
- MAP@100
- latency p50 / p95
- QPS

This script assumes:
- IVF-HNSW index already built
- FlatIP oracle index already built
- oracle_topk.npz may already exist (will reuse if present)
"""

import os
import time
import numpy as np
import faiss
from typing import Dict, List
from dataclasses import dataclass

# ----------------------------
# PATHS
# ----------------------------
INDEX_DIR = os.path.expanduser("~/data/indexes_full")
CACHE_DIR = os.path.join(INDEX_DIR, "eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

IVF_HNSW_INDEX_PATH = os.path.join(
    INDEX_DIR, "faiss_ivf_hnsw_ip_nlist16384_M32.index"
)
FLAT_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_flat_ip.index")

ORACLE_CACHE_PATH = os.path.join(CACHE_DIR, "oracle_topk.npz")
OUT_CSV = os.path.join(CACHE_DIR, "results_ivf_hnsw_eval.csv")

DATA_DIR = os.path.expanduser("~/data")
MSMARCO_DIR = os.path.join(DATA_DIR, "msmarco")

QUERY_EMB_PATH = os.path.join(DATA_DIR, "embeddings", "query_emb_dev.npy")
DEV_QIDS_PATH = os.path.join(MSMARCO_DIR, "dev_query_ids.txt")
QRELS_PATH = os.path.join(MSMARCO_DIR, "qrels.dev.tsv")
DOCIDS_PATH = os.path.join(MSMARCO_DIR, "doc_ids.txt")

# ----------------------------
# EVAL PARAMS
# ----------------------------
CANDIDATE_K = 200
ORACLE_K = 200
NDCG_K = 10
MRR_K = 10
MAP_K = 100

NPROBE_SWEEP = [4, 8, 16, 32]
EF_SEARCH_SWEEP = [32, 64, 128]

QUERY_BATCH = 256

# ----------------------------
# HELPERS
# ----------------------------
def load_qids(path):
    with open(path, "r", encoding="utf-8") as f:
        return np.array([l.strip() for l in f if l.strip()], dtype=object)

def load_docids(path):
    with open(path, "r", encoding="utf-8") as f:
        return np.array([l.strip() for l in f if l.strip()], dtype=object)

def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split("\t")
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels

def dcg_at_k(rels, k):
    rels = rels[:k]
    if len(rels) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum((2 ** rels - 1) * discounts))

def ndcg_at_k(ranked, qrel, k):
    rels = np.array([qrel.get(d, 0) for d in ranked[:k]], dtype=np.float32)
    dcg = dcg_at_k(rels, k)
    ideal = np.array(sorted(qrel.values(), reverse=True), dtype=np.float32)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def mrr_at_k(ranked, qrel, k):
    for i, d in enumerate(ranked[:k], 1):
        if qrel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def ap_at_k(ranked, qrel, k):
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if qrel.get(d, 0) > 0:
            hits += 1
            score += hits / i
    return 0.0 if hits == 0 else score / hits

def recall_at_k(approx, oracle, k):
    return len(set(approx[:k]) & set(oracle[:k])) / len(oracle[:k])

def percentile(x, p):
    return float(np.percentile(x, p))

# ----------------------------
# FAISS SEARCH
# ----------------------------
@dataclass
class SearchResult:
    ids: np.ndarray
    lat_ms: np.ndarray

def batched_search(index, queries, k):
    nq = queries.shape[0]
    ids = np.empty((nq, k), dtype=np.int64)
    lat = np.empty(nq, dtype=np.float32)

    for s in range(0, nq, QUERY_BATCH):
        e = min(s + QUERY_BATCH, nq)
        t0 = time.perf_counter()
        _, I = index.search(queries[s:e], k)
        t1 = time.perf_counter()
        ids[s:e] = I
        lat[s:e] = (t1 - t0) * 1000.0 / (e - s)

    return SearchResult(ids, lat)

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Load data
    queries = np.load(QUERY_EMB_PATH, mmap_mode="r").astype(np.float32)
    qids = load_qids(DEV_QIDS_PATH)
    docids = load_docids(DOCIDS_PATH)
    qrels = load_qrels(QRELS_PATH)

    # Load indexes
    ivf_hnsw = faiss.read_index(IVF_HNSW_INDEX_PATH)
    flat = faiss.read_index(FLAT_INDEX_PATH)

    faiss.omp_set_num_threads(faiss.omp_get_max_threads())

    # Oracle
    if os.path.exists(ORACLE_CACHE_PATH):
        oracle_ids = np.load(ORACLE_CACHE_PATH)["oracle_ids"]
    else:
        oracle = batched_search(flat, queries, ORACLE_K)
        oracle_ids = oracle.ids
        np.savez_compressed(ORACLE_CACHE_PATH, oracle_ids=oracle_ids)

    rows = []

    # IMPORTANT FIX: downcast quantizer once
    hnsw_quantizer = faiss.downcast_index(ivf_hnsw.quantizer)

    for nprobe in NPROBE_SWEEP:
        ivf_hnsw.nprobe = nprobe
        for ef in EF_SEARCH_SWEEP:
            hnsw_quantizer.hnsw.efSearch = ef

            res = batched_search(ivf_hnsw, queries, CANDIDATE_K)

            recalls, ndcgs, mrrs, aps = [], [], [], []

            for i in range(len(qids)):
                qrel = qrels.get(str(qids[i]), {})
                approx = res.ids[i]
                oracle = oracle_ids[i]

                recalls.append(recall_at_k(approx, oracle, ORACLE_K))
                ranked = [docids[int(d)] for d in approx if d >= 0]

                ndcgs.append(ndcg_at_k(ranked, qrel, NDCG_K))
                mrrs.append(mrr_at_k(ranked, qrel, MRR_K))
                aps.append(ap_at_k(ranked, qrel, MAP_K))

            rows.append([
                "ivf_hnsw", nprobe, ef,
                np.mean(recalls),
                np.mean(ndcgs),
                np.mean(mrrs),
                np.mean(aps),
                percentile(res.lat_ms, 50),
                percentile(res.lat_ms, 95),
                1000.0 / np.mean(res.lat_ms)
            ])

            print(f"nprobe={nprobe} ef={ef} nDCG@10={np.mean(ndcgs):.4f}")

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write(
            "method,nprobe,efSearch,recall@200,ndcg@10,mrr@10,map@100,"
            "lat_p50_ms,lat_p95_ms,qps\n"
        )
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()

