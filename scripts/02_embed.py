"""
02_embed.py

Embeds MS MARCO documents and queries using SentenceTransformers.
Uses Apple Silicon GPU via PyTorch MPS when available, otherwise CPU.

Inputs:
- data/raw/documents.json
- data/raw/queries.json

Outputs:
- data/embeddings/doc_ids.json
- data/embeddings/query_ids.json
- data/embeddings/doc_shards.json
- data/embeddings/doc_emb_000.npy, doc_emb_001.npy, ...
- data/embeddings/query_emb.npy
"""

import os
import json
from typing import List, Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Apple Silicon GPUs prefer smaller batches than CUDA GPUs
BATCH_SIZE_MPS = 64
BATCH_SIZE_CPU = 32

DOC_SHARD_SIZE = 50_000   # 500k docs -> 10 shards
DTYPE = np.float32


def chunk_range(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        yield start, end


def pick_device_and_batch():
    import torch
    if torch.backends.mps.is_available():
        return "mps", BATCH_SIZE_MPS
    return "cpu", BATCH_SIZE_CPU


def main():
    docs_path = os.path.join(RAW_DIR, "documents.json")
    queries_path = os.path.join(RAW_DIR, "queries.json")

    print("Loading documents.json ...")
    with open(docs_path, "r") as f:
        docs: List[Dict] = json.load(f)

    print("Loading queries.json ...")
    with open(queries_path, "r") as f:
        queries: List[Dict] = json.load(f)

    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["text"] for q in queries]

    # Save IDs for later mapping
    with open(os.path.join(OUT_DIR, "doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)
    with open(os.path.join(OUT_DIR, "query_ids.json"), "w") as f:
        json.dump(query_ids, f)

    device, batch_size = pick_device_and_batch()
    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device} | batch size: {batch_size}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=device)

    # ---- Embed documents in shards ----
    n_docs = len(doc_texts)
    ranges = list(chunk_range(n_docs, DOC_SHARD_SIZE))
    print(f"Embedding {n_docs} documents in {len(ranges)} shards")

    shard_files = []

    for shard_id, (start, end) in enumerate(tqdm(ranges)):
        out_path = os.path.join(OUT_DIR, f"doc_emb_{shard_id:03d}.npy")
        shard_files.append(out_path)

        if os.path.exists(out_path):
            continue  # resume support

        texts = doc_texts[start:end]
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype(DTYPE)

        np.save(out_path, emb)

    with open(os.path.join(OUT_DIR, "doc_shards.json"), "w") as f:
        json.dump(shard_files, f, indent=2)

    # ---- Embed queries ----
    print(f"Embedding {len(query_texts)} queries")
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(DTYPE)

    np.save(os.path.join(OUT_DIR, "query_emb.npy"), query_emb)

    print("Done. Embeddings saved in data/embeddings/")


if __name__ == "__main__":
    main()
