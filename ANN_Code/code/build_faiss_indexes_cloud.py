"""
I build FAISS ANN indices from my MS MARCO passage embeddings.

What I verified about my data:
- 45 shards: ~/data/embeddings/doc_emb_*.npy
- dtype: float32 for all shards
- shape: (<=200000, 768) per shard
- total vectors: 8,841,823
- vectors are L2-normalized (norm ~ 1.0)

Therefore:
- cosine similarity == inner product
- I use INNER PRODUCT indices everywhere (IndexFlatIP, METRIC_INNER_PRODUCT, etc.)
- I stream shards one-by-one to keep memory stable.

Indices I build:
1) FlatIP (exact baseline)
2) IVF-Flat IP (scalable)
3) IVF-PQ IP (compressed)
4) HNSW IP (graph-based)
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import faiss

# --------------------------
# Paths
# --------------------------
EMB_DIR = Path("~/data/embeddings").expanduser()
OUT_DIR = Path("~/data/indexes_full").expanduser()
LOG_DIR = OUT_DIR / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# Parameters (override with env vars)
# --------------------------
FAISS_THREADS = int(os.environ.get("FAISS_THREADS", "32"))

# Train IVF/PQ on a sample (standard practice)
TRAIN_N = int(os.environ.get("TRAIN_N", "500000"))

# IVF
IVF_NLIST = int(os.environ.get("IVF_NLIST", "16384"))

# IVF-PQ
PQ_M = int(os.environ.get("PQ_M", "64"))
PQ_NBITS = int(os.environ.get("PQ_NBITS", "8"))

# HNSW
HNSW_M = int(os.environ.get("HNSW_M", "32"))
HNSW_EFC = int(os.environ.get("HNSW_EFC", "200"))


def shard_files():
    files = sorted(EMB_DIR.glob("doc_emb_*.npy"))
    if not files:
        raise FileNotFoundError(f"No shards found in {EMB_DIR}")
    return files


def infer_dim():
    x = np.load(shard_files()[0], mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"Unexpected shard shape: {x.shape}")
    return x.shape[1]


def count_total_vectors_and_check():
    """
    I scan all shards to confirm:
    - consistent dimension
    - consistent dtype float32
    - total vector count
    """
    files = shard_files()
    dset = set()
    dtypes = set()
    total = 0

    for f in files:
        x = np.load(f, mmap_mode="r")
        if x.ndim != 2:
            raise ValueError(f"{f.name} has shape {x.shape}")
        dset.add(x.shape[1])
        dtypes.add(str(x.dtype))
        total += x.shape[0]

    if len(dset) != 1:
        raise ValueError(f"Inconsistent dims across shards: {dset}")
    if dtypes != {"float32"}:
        raise ValueError(f"Unexpected dtypes found: {dtypes} (expected only float32)")

    return next(iter(dset)), total


def iter_shards():
    """
    I stream shards one-by-one as float32 arrays (FAISS requires float32).
    Using mmap keeps memory stable.
    """
    for f in shard_files():
        x = np.load(f, mmap_mode="r")
        # x is already float32, but I keep this explicit.
        yield f.name, x.astype("float32", copy=False)


def sample_training_vectors(max_train=TRAIN_N, seed=123):
    """
    I sample vectors across shards for IVF/PQ training.
    Training on all 8.8M vectors is unnecessary and slow.
    """
    rng = np.random.default_rng(seed)
    files = shard_files()
    rng.shuffle(files)

    d = infer_dim()
    out = np.empty((max_train, d), dtype="float32")
    filled = 0

    for f in files:
        if filled >= max_train:
            break

        x = np.load(f, mmap_mode="r").astype("float32", copy=False)
        n = x.shape[0]

        # I cap sample per shard so one shard doesn't dominate.
        take = min(max_train - filled, n, 200_000)
        idx = rng.choice(n, size=take, replace=False)
        out[filled:filled + take] = x[idx]
        filled += take

    return out[:filled]


def write_meta(name, meta):
    """
    I write metadata logs so I can reproduce every index exactly.
    """
    path = LOG_DIR / f"{name}.meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print("📝 wrote", path)


def add_all(index):
    """
    I add all vectors to an index shard-by-shard.
    """
    total = 0
    t0 = time.time()

    for shard_name, x in iter_shards():
        index.add(x)
        total += x.shape[0]

        if total % 1_000_000 == 0:
            print(f"  added {total:,} vectors | elapsed {time.time()-t0:.1f}s")

    return total, time.time() - t0


def build_flat_ip(d):
    name = "faiss_flat_ip"
    path = OUT_DIR / f"{name}.index"
    if path.exists():
        print(f"✅ Found {path.name}, skipping")
        return

    print(f"⏳ Building {name} (exact)")
    t0 = time.time()

    index = faiss.IndexFlatIP(d)
    n_added, add_s = add_all(index)
    faiss.write_index(index, str(path))

    write_meta(name, {
        "name": name,
        "type": "IndexFlatIP",
        "metric": "INNER_PRODUCT",
        "dim": d,
        "vectors_added": n_added,
        "build_seconds": time.time() - t0,
        "add_seconds": add_s,
        "threads": FAISS_THREADS,
    })
    print(f"✅ Done {name}")


def build_ivf_flat_ip(d):
    name = f"faiss_ivf_flat_ip_nlist{IVF_NLIST}"
    path = OUT_DIR / f"{name}.index"
    if path.exists():
        print(f"✅ Found {path.name}, skipping")
        return

    print(f"⏳ Building {name}")
    t0 = time.time()

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)

    train = sample_training_vectors()
    print("  training on", train.shape)
    index.train(train)

    n_added, add_s = add_all(index)
    faiss.write_index(index, str(path))

    write_meta(name, {
        "name": name,
        "type": "IndexIVFFlat",
        "metric": "INNER_PRODUCT",
        "dim": d,
        "nlist": IVF_NLIST,
        "train_n": int(train.shape[0]),
        "vectors_added": n_added,
        "build_seconds": time.time() - t0,
        "add_seconds": add_s,
        "threads": FAISS_THREADS,
    })
    print(f"✅ Done {name}")


def build_ivf_pq_ip(d):
    name = f"faiss_ivf_pq_ip_nlist{IVF_NLIST}_m{PQ_M}_b{PQ_NBITS}"
    path = OUT_DIR / f"{name}.index"
    if path.exists():
        print(f"✅ Found {path.name}, skipping")
        return

    print(f"⏳ Building {name}")
    t0 = time.time()

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(
        quantizer, d, IVF_NLIST, PQ_M, PQ_NBITS, faiss.METRIC_INNER_PRODUCT
    )

    train = sample_training_vectors()
    print("  training on", train.shape)
    index.train(train)

    n_added, add_s = add_all(index)
    faiss.write_index(index, str(path))

    write_meta(name, {
        "name": name,
        "type": "IndexIVFPQ",
        "metric": "INNER_PRODUCT",
        "dim": d,
        "nlist": IVF_NLIST,
        "m": PQ_M,
        "nbits": PQ_NBITS,
        "train_n": int(train.shape[0]),
        "vectors_added": n_added,
        "build_seconds": time.time() - t0,
        "add_seconds": add_s,
        "threads": FAISS_THREADS,
    })
    print(f"✅ Done {name}")


def build_hnsw_ip(d):
    name = f"faiss_hnsw_ip_M{HNSW_M}_efC{HNSW_EFC}"
    path = OUT_DIR / f"{name}.index"
    if path.exists():
        print(f"✅ Found {path.name}, skipping")
        return

    print(f"⏳ Building {name}")
    t0 = time.time()

    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EFC

    n_added, add_s = add_all(index)
    faiss.write_index(index, str(path))

    write_meta(name, {
        "name": name,
        "type": "IndexHNSWFlat",
        "metric": "INNER_PRODUCT",
        "dim": d,
        "M": HNSW_M,
        "efConstruction": HNSW_EFC,
        "vectors_added": n_added,
        "build_seconds": time.time() - t0,
        "add_seconds": add_s,
        "threads": FAISS_THREADS,
    })
    print(f"✅ Done {name}")


def main():
    # I fix the number of threads to make runtimes comparable across runs.
    os.environ["OMP_NUM_THREADS"] = str(FAISS_THREADS)
    faiss.omp_set_num_threads(FAISS_THREADS)

    d, total = count_total_vectors_and_check()

    print("\n========== DATA SUMMARY ==========")
    print("Shards        :", len(shard_files()))
    print("Embedding dim :", d)
    print("Total vectors :", f"{total:,}")
    print("FAISS threads :", FAISS_THREADS)
    print("=================================\n")

    # I build indices in an order that gives me useful baselines early.
    build_flat_ip(d)
    build_ivf_flat_ip(d)
    #build_ivf_pq_ip(d)
    build_hnsw_ip(d)

    print("\n🎉 All FAISS indices built successfully.")
    print("Indices are in:", OUT_DIR)


if __name__ == "__main__":
    main()

