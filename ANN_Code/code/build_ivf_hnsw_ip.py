# build_ivf_hnsw_ip.py
"""
Build an IVF-HNSW index for MS MARCO document embeddings.

Why IVF-HNSW:
- IVF limits the search space (coarse quantization)
- HNSW provides fast graph-based navigation
- Architecturally similar to DiskANN
- Fully supported in FAISS (no MKL / Docker pain)
"""

import os
import time
import faiss
import numpy as np
from pathlib import Path

# --------------------------
# Paths (reuse your structure)
# --------------------------
EMB_DIR = Path("~/data/embeddings").expanduser()
OUT_DIR = Path("~/data/indexes_full").expanduser()
LOG_DIR = OUT_DIR / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Parameters
# --------------------------
FAISS_THREADS = 32

DIM = 768
IVF_NLIST = 16384          # same as IVF-Flat
HNSW_M = 32                # graph degree
HNSW_EF_CONSTR = 200       # build accuracy

INDEX_NAME = f"faiss_ivf_hnsw_ip_nlist{IVF_NLIST}_M{HNSW_M}"
INDEX_PATH = OUT_DIR / f"{INDEX_NAME}.index"
META_PATH = LOG_DIR / f"{INDEX_NAME}.meta.json"

# --------------------------
# Helpers (reuse logic)
# --------------------------
def shard_files():
    return sorted(EMB_DIR.glob("doc_emb_*.npy"))

def iter_shards():
    for f in shard_files():
        x = np.load(f, mmap_mode="r")
        yield f.name, x.astype("float32", copy=False)

def sample_training_vectors(max_train=500_000, seed=123):
    rng = np.random.default_rng(seed)
    files = shard_files()
    rng.shuffle(files)

    out = np.empty((max_train, DIM), dtype="float32")
    filled = 0

    for f in files:
        if filled >= max_train:
            break
        x = np.load(f, mmap_mode="r")
        take = min(max_train - filled, x.shape[0], 200_000)
        idx = rng.choice(x.shape[0], size=take, replace=False)
        out[filled:filled + take] = x[idx]
        filled += take

    return out[:filled]

# --------------------------
# Build IVF-HNSW
# --------------------------
def main():
    os.environ["OMP_NUM_THREADS"] = str(FAISS_THREADS)
    faiss.omp_set_num_threads(FAISS_THREADS)

    if INDEX_PATH.exists():
        print("Index already exists, skipping:", INDEX_PATH)
        return

    print("Building IVF-HNSW index")
    t0 = time.time()

    # HNSW quantizer instead of Flat
    quantizer = faiss.IndexHNSWFlat(DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = HNSW_EF_CONSTR

    # IVF index using HNSW quantizer
    index = faiss.IndexIVFFlat(
        quantizer,
        DIM,
        IVF_NLIST,
        faiss.METRIC_INNER_PRODUCT
    )

    # Train IVF
    train = sample_training_vectors()
    print("Training on", train.shape)
    index.train(train)

    # Add vectors shard-by-shard
    total = 0
    for _, x in iter_shards():
        index.add(x)
        total += x.shape[0]
        if total % 1_000_000 == 0:
            print(f"  added {total:,} vectors")

    faiss.write_index(index, str(INDEX_PATH))

    # Metadata for reproducibility
    meta = {
        "name": INDEX_NAME,
        "type": "IndexIVFFlat + HNSW quantizer",
        "metric": "INNER_PRODUCT",
        "dim": DIM,
        "nlist": IVF_NLIST,
        "hnsw_M": HNSW_M,
        "efConstruction": HNSW_EF_CONSTR,
        "vectors_added": total,
        "build_seconds": time.time() - t0,
        "threads": FAISS_THREADS
    }

    import json
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done:", INDEX_PATH)

if __name__ == "__main__":
    main()

