"""
03_build_indexes.py

Builds FAISS indexes (Inner Product) from sharded doc embeddings.

Inputs:
- data/embeddings/doc_shards.json
- data/embeddings/doc_emb_*.npy

Outputs:
- indexes/faiss_hnsw_ip.index
- indexes/faiss_ivf_flat_ip.index
- indexes/faiss_ivf_pq_ip.index
- indexes/meta.json  (index config + counts)

Notes:
- Embeddings were already L2-normalized, so IP == cosine similarity.
"""

import os
import json
import time
from typing import List

import numpy as np

INDEX_DIR = "indexes"
EMB_DIR = "data/embeddings"
os.makedirs(INDEX_DIR, exist_ok=True)

DOC_SHARDS_PATH = os.path.join(EMB_DIR, "doc_shards.json")

# -------------------------
# Config
# -------------------------
DIM = 768

# HNSW params
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# IVF params
IVF_NLIST = 4096          # coarse clusters (adjustable)
IVF_TRAIN_SIZE = 200_000  # number of vectors used to train IVF/PQ
PQ_M = 64                 # number of subquantizers (must divide DIM)
PQ_BITS = 8               # bits per subvector code

# -------------------------
def load_shards_list() -> List[str]:
    with open(DOC_SHARDS_PATH, "r") as f:
        return json.load(f)

def iter_vectors(shard_paths: List[str]):
    for p in shard_paths:
        x = np.load(p)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        yield x

def sample_for_training(shard_paths: List[str], train_size: int) -> np.ndarray:
    """Uniformly sample vectors across shards for training."""
    # Load progressively until we have enough
    samples = []
    remaining = train_size
    for p in shard_paths:
        x = np.load(p).astype(np.float32)
        if remaining <= 0:
            break
        take = min(len(x), max(1, remaining))
        # random sample within shard (but deterministic-ish by fixed seed)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=take, replace=False) if take < len(x) else np.arange(len(x))
        samples.append(x[idx])
        remaining -= take
    return np.vstack(samples)

def add_all_vectors(index, shard_paths: List[str]) -> int:
    total = 0
    for x in iter_vectors(shard_paths):
        index.add(x)
        total += x.shape[0]
        print(f"  added {x.shape[0]} (total {total})")
    return total

def main():
    import faiss

    shard_paths = load_shards_list()
    print(f"Found {len(shard_paths)} doc shards")
    print("Example shard:", shard_paths[0])

    # -------------------------
    # HNSW (IP)
    # -------------------------
    hnsw_path = os.path.join(INDEX_DIR, "faiss_hnsw_ip.index")
    if not os.path.exists(hnsw_path):
        print("\nBuilding FAISS HNSW (IP)...")
        t0 = time.time()
        hnsw = faiss.IndexHNSWFlat(DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        # Add vectors
        total = add_all_vectors(hnsw, shard_paths)
        faiss.write_index(hnsw, hnsw_path)
        print(f"HNSW done. vectors={total} time={time.time()-t0:.1f}s saved={hnsw_path}")
    else:
        print(f"\nHNSW index exists: {hnsw_path}")

    # -------------------------
    # IVF-Flat (IP)
    # -------------------------
    ivf_flat_path = os.path.join(INDEX_DIR, "faiss_ivf_flat_ip.index")
    if not os.path.exists(ivf_flat_path):
        print("\nTraining FAISS IVF-Flat (IP)...")
        t0 = time.time()
        quantizer = faiss.IndexFlatIP(DIM)
        ivf = faiss.IndexIVFFlat(quantizer, DIM, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)

        train_x = sample_for_training(shard_paths, IVF_TRAIN_SIZE)
        print("Training samples shape:", train_x.shape)
        ivf.train(train_x)

        print("Adding vectors to IVF-Flat...")
        total = add_all_vectors(ivf, shard_paths)
        faiss.write_index(ivf, ivf_flat_path)
        print(f"IVF-Flat done. vectors={total} time={time.time()-t0:.1f}s saved={ivf_flat_path}")
    else:
        print(f"\nIVF-Flat index exists: {ivf_flat_path}")

    # -------------------------
    # IVF-PQ (IP)
    # -------------------------
    ivf_pq_path = os.path.join(INDEX_DIR, "faiss_ivf_pq_ip.index")
    if not os.path.exists(ivf_pq_path):
        print("\nTraining FAISS IVF-PQ (IP)...")
        t0 = time.time()
        quantizer = faiss.IndexFlatIP(DIM)
        ivfpq = faiss.IndexIVFPQ(
            quantizer, DIM, IVF_NLIST, PQ_M, PQ_BITS, faiss.METRIC_INNER_PRODUCT
        )

        train_x = sample_for_training(shard_paths, IVF_TRAIN_SIZE)
        print("Training samples shape:", train_x.shape)
        ivfpq.train(train_x)

        print("Adding vectors to IVF-PQ...")
        total = add_all_vectors(ivfpq, shard_paths)
        faiss.write_index(ivfpq, ivf_pq_path)
        print(f"IVF-PQ done. vectors={total} time={time.time()-t0:.1f}s saved={ivf_pq_path}")
    else:
        print(f"\nIVF-PQ index exists: {ivf_pq_path}")

    meta = {
        "dim": DIM,
        "num_docs": 500000,
        "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
        "ivf": {"nlist": IVF_NLIST, "train_size": IVF_TRAIN_SIZE},
        "pq": {"m": PQ_M, "bits": PQ_BITS},
    }
    with open(os.path.join(INDEX_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nAll indexes built. Meta saved to indexes/meta.json")

if __name__ == "__main__":
    main()
