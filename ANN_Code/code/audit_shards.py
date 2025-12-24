from pathlib import Path
import numpy as np

emb_dir = Path("~/data/embeddings").expanduser()
files = sorted(emb_dir.glob("doc_emb_*.npy"))

dtypes = {}
dims = set()
total = 0

for f in files:
    x = np.load(f, mmap_mode="r")
    dtypes[str(x.dtype)] = dtypes.get(str(x.dtype), 0) + 1
    dims.add(x.shape[1])
    total += x.shape[0]

print("num_shards:", len(files))
print("dtype_counts:", dtypes)
print("dims_found:", dims)
print("total_vectors:", total)

