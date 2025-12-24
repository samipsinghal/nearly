import json, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
QUERY_JSON = os.path.expanduser("~/data/msmarco/queries.json")
OUT_EMB    = os.path.expanduser("~/data/embeddings/query_emb_dev.npy")
OUT_QIDS   = os.path.expanduser("~/data/msmarco/dev_query_ids.txt")
BATCH_SIZE = 256

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer(MODEL_NAME, device=device)

    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    qids  = [str(x["id"]) for x in data]
    texts = [x["text"] for x in data]

    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(OUT_EMB, emb)

    with open(OUT_QIDS, "w", encoding="utf-8") as f:
        f.write("\n".join(qids) + "\n")

    print("Saved:", OUT_EMB, emb.shape, emb.dtype)
    print("Saved:", OUT_QIDS, len(qids))

if __name__ == "__main__":
    main()

