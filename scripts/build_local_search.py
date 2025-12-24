import os
import glob
import numpy as np
import faiss
import json
from google.cloud import storage
from tqdm import tqdm

# --- CONFIG ---
BUCKET_NAME = "nearly-search-data"  # <--- CHANGE THIS
EMB_DIR = "data_vectors"
INDEX_DIR = "data_index"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def main():
    print("--- 3. BUILDING INDEX ON M4 MAX ---")
    
    # 1. Download Vectors
    client = storage.Client()
    blobs = list(client.bucket(BUCKET_NAME).list_blobs(prefix="embeddings/"))
    print(f"Syncing {len(blobs)} files from Cloud...")
    
    local_files = []
    for blob in tqdm(blobs):
        path = os.path.join(EMB_DIR, os.path.basename(blob.name))
        if not os.path.exists(path):
            blob.download_to_file(open(path, "wb"))
        if path.endswith(".npy") and "doc_emb" in path:
            local_files.append(path)
    local_files.sort()

    # 2. Train Compressed Index (IVF-PQ)
    # We use a sample because 27GB won't fit for training
    print("Training Index (Compressing)...")
    train_data = np.load(local_files[0]) # Load just first shard (200k docs)
    d = train_data.shape[1] # 768
    
    # IVF16384 = Fast search (clusters data)
    # PQ64 = High compression (reduces vector size by 12x)
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, 16384, 64, 8)
    
    # Train on the M4 Max CPU (It's fast!)
    index.train(train_data)
    
    # 3. Add All Data
    print("Adding data to index...")
    for f in tqdm(local_files):
        # We load one small shard, add it, then delete it from RAM.
        # This keeps RAM usage very low.
        shard = np.load(f)
        index.add(shard)

    # 4. Save
    faiss.write_index(index, os.path.join(INDEX_DIR, "msmarco.index"))
    print("✅ SUCCESS! Search engine is ready.")

if __name__ == "__main__":
    main()