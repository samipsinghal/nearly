import json
import os
import numpy as np
import torch
from google.cloud import storage
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
BUCKET_NAME = "nearly-search-data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# NOW WE CAN GO BIG
BATCH_SIZE = 2048   # L4 can handle this easily with enough RAM
SHARD_SIZE = 200_000

def main():
    print("--- RUNNING ON G2-STANDARD-16 (TURBO MODE) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 1. Load Data
    if not os.path.exists("documents.json"):
        print("Downloading documents.json...")
        blob = bucket.blob("raw/documents.json")
        blob.download_to_filename("documents.json")

    print("Loading JSON (This is safe now with 64GB RAM)...")
    with open("documents.json") as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs]

    # 2. Setup Model & Pool
    print(f"Loading Model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # CRITICAL: Start the multi-process pool on your 16 CPUs
    print("Starting Multi-Process Pool...")
    pool = model.start_multi_process_pool()

    total = len(texts)
    print(f"Embedding {total} documents...")

    for i in range(0, total, SHARD_SIZE):
        shard_name = f"doc_emb_{i//SHARD_SIZE:03d}.npy"
        remote_path = f"embeddings/{shard_name}"

        if bucket.blob(remote_path).exists():
            print(f"Skipping {shard_name} (already done).")
            continue

        print(f"Processing shard {i//SHARD_SIZE}...")
        batch = texts[i : i + SHARD_SIZE]
        
        # USE THE POOL
        emb = model.encode_multi_process(
            batch, 
            pool, 
            batch_size=BATCH_SIZE
        )
        
        # Save & Upload
        np.save(shard_name, emb.astype(np.float32))
        print(f"Uploading {shard_name}...")
        bucket.blob(remote_path).upload_from_filename(shard_name)
        os.remove(shard_name)

    # Cleanup
    print("Stopping Pool...")
    model.stop_multi_process_pool(pool)
    print("Job Done.")

if __name__ == "__main__":
    main()
