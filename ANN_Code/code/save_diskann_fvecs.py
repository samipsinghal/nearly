# save_diskann_fvecs.py
"""
Convert MS MARCO document embeddings into DiskANN fvecs format.

Why:
- DiskANN expects fvecs, not .npy
- Each vector is stored as:
    [int32 dimension][float32 * dimension]

This is a ONE-TIME preprocessing step.
"""

import numpy as np
from pathlib import Path

# Directory containing FAISS shards (doc_emb_*.npy)
EMB_DIR = Path("~/data/embeddings").expanduser()

# Output directory for DiskANN data
OUT_DIR = Path("~/data/diskann").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "doc_vectors.fvecs"

def write_fvecs(path, X):
    """
    Write vectors to DiskANN fvecs format.
    """
    with open(path, "wb") as f:
        for v in X:
            # Write vector dimension
            f.write(np.int32(len(v)).tobytes())
            # Write vector values
            f.write(v.astype(np.float32).tobytes())

# Load shards one by one to avoid memory explosion
chunks = []
for shard in sorted(EMB_DIR.glob("doc_emb_*.npy")):
    x = np.load(shard, mmap_mode="r")
    chunks.append(x)

# Concatenate all shards
X = np.vstack(chunks)

write_fvecs(OUT_FILE, X)

print(f"Wrote DiskANN fvecs file: {OUT_FILE}")
print("Shape:", X.shape)

