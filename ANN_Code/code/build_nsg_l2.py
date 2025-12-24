#!/usr/bin/env python3
"""
Build NSG (Navigating Spreading-out Graph) index using NMSLIB with L2 distance.

IMPORTANT:
- NMSLIB NSG does NOT support cosine / inner product.
- Since all embeddings are L2-normalized, L2 distance is equivalent
  to cosine similarity up to a monotonic transformation.
- This is scientifically correct and commonly done.

Index type:
- Single-layer graph ANN (contrast with HNSW hierarchy)
"""

import time
import json
import numpy as np
import nmslib
from pathlib import Path

# --------------------------
# Paths
# --------------------------
EMB_DIR = Path("~/data/embeddings").expanduser()
OUT_DIR = Path("~/data/indexes_full").expanduser()
LOG_DIR = OUT_DIR / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "nmslib_nsg_l2"
INDEX_PATH = OUT_DIR / INDEX_NAME
META_PATH = LOG_DIR / f"{INDEX_NAME}.meta.json"

# --------------------------
# NSG parameters
# --------------------------
NSG_M = 32
NSG_EF_CONSTR = 200

# --------------------------
# Helpers
# --------------------------
def iter_shards():
    """
    Stream document embedding shards one-by-one.
    Embeddings are already float32 and L2-normalized.
    """
    for f in sorted(EMB_DIR.glob("doc_emb_*.npy")):
        x = np.load(f, mmap_mode="r")
        yield x.astype("float32", copy=False)

# --------------------------
# Build NSG
# --------------------------
def main():
    if INDEX_PATH.exists():
        print("Index already exists, skipping:", INDEX_PATH)
        return

    print("Building NSG (L2) index")
    t0 = time.time()

    # NSG only supports L2
    index = nmslib.init(
        method="nsg",
        space="l2"
    )

    total = 0
    for x in iter_shards():
        index.addDataPointBatch(x)
        total += x.shape[0]
        if total % 1_000_000 == 0:
            print(f"  added {total:,} vectors")

    # Build the graph
    index.createIndex(
        {
            "M": NSG_M,
            "efConstruction": NSG_EF_CONSTR
        },
        print_progress=True
    )

    # Persist index to disk
    index.saveIndex(str(INDEX_PATH), save_data=True)

    meta = {
        "name": INDEX_NAME,
        "type": "NSG (single-layer graph)",
        "distance": "L2 (embeddings normalized)",
        "M": NSG_M,
        "efConstruction": NSG_EF_CONSTR,
        "vectors": total,
        "build_seconds": time.time() - t0
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("Done:", INDEX_PATH)

if __name__ == "__main__":
    main()

