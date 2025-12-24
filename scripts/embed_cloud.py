import json
import os
import numpy as np
import torch
from google.cloud import storage
from sentence_transformers import SentenceTransformer

# --- CONFIG FOR G2-STANDARD-32 (FP16 MODE) ---
BUCKET_NAME = "nearly-search-data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# 1024 fits easily in VRAM when using FP16
BATCH_SIZE = 1024  
SHARD_SIZE = 200_000

def main():
    print("--- RUNNING ON FP16 HYPER-SPEED MODE ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 1. Load Data
    if not os.path.exists("documents.json"):
        print("Downloading documents.json...")
        blob = bucket.blob("raw/documents.json")
        blob.download_to_filename("documents.json")

    print("Loading JSON...")
    with open("documents.json") as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs]
    
    # 2. Setup Model
    print(f"Loading Model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # CRITICAL OPTIMIZATION: Switch to Half Precision (FP16)
    # This activated the L4 Tensor Cores for massive speedup
    print("Switching model to FP16 (Half Precision)...")
    model.half() 
    
    total = len(texts)
    print(f"Embedding {total} documents with Batch Size {BATCH_SIZE}...")

    for i in range(0, total, SHARD_SIZE):
        shard_name = f"doc_emb_{i//SHARD_SIZE:03d}.npy"
        remote_path = f"embeddings/{shard_name}"

        # Resume Logic: Check if file already exists in Cloud
        if bucket.blob(remote_path).exists():
            print(f"Skipping {shard_name} (already done).")
            continue

        print(f"Processing shard {i//SHARD_SIZE} ({i} to {i+SHARD_SIZE})...")
        batch = texts[i : i + SHARD_SIZE]
        
        # Encode with FP16
        emb = model.encode(
            batch, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # Save & Upload (Casting back to float32 for standard NPY storage)
        np.save(shard_name, emb.astype(np.float32))
        print(f"Uploading {shard_name}...")
        bucket.blob(remote_path).upload_from_filename(shard_name)
        
        # Clean up local disk space
        os.remove(shard_name)

    print("Job Done.")

if __name__ == "__main__":
    main()