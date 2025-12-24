import numpy as np
from pathlib import Path

p = Path("~/data/embeddings/doc_emb_000.npy").expanduser()
x = np.load(p, mmap_mode="r")[:20000].astype("float32")

norms = np.linalg.norm(x, axis=1)
print("min:", norms.min())
print("max:", norms.max())
print("mean:", norms.mean())

