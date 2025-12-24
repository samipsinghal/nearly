import json
import os
import numpy as np
import torch
from google.cloud import storage
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
BUCKET_NAME = "nearly-search-data"  # <--- CHANGE THIS
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 256  # Good for T4 GPU
SHARD_SIZE = 200_000

def main():
    print("--- 2. RUNNING ON CLOUD GPU ---")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 1. Download Data
    if not os.path.exists("documents.json"):
        print("Downloading documents.json...")
        bucket.blob("raw/documents.json").download_to_filename("documents.json")

    print("Loading JSON...")
    with open("documents.json") as f:
        docs = json.load(f)
    texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]

    # Save IDs separately (we need these later!)
    with open("doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    bucket.blob("embeddings/doc_ids.json").upload_from_filename("doc_ids.json")

    # 2. Embed Loop
    print(f"Loading Model on {torch.cuda.get_device_name(0)}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    
    total = len(texts)
    print(f"Embedding {total} documents...")

    for i in range(0, total, SHARD_SIZE):
        shard_name = f"doc_emb_{i//SHARD_SIZE:03d}.npy"
        remote_path = f"embeddings/{shard_name}"

        # Check if exists (Resume capability)
        if bucket.blob(remote_path).exists():
            print(f"Skipping {shard_name} (already done).")
            continue

        # Embed
        batch = texts[i : i + SHARD_SIZE]
        emb = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
        
        # Save & Upload
        np.save(shard_name, emb.astype(np.float32))
        bucket.blob(remote_path).upload_from_filename(shard_name)
        os.remove(shard_name)  # Clear disk space

    print("✅ Job Done. You can delete this VM now.")

if __name__ == "__main__":
    main()