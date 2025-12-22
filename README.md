# NEARLY  
**NEAR-neighbor Retrieval anaLYsis**  
*A ranking-centric benchmark of Approximate Nearest Neighbor (ANN) methods for dense retrieval on MS MARCO*

---

## 1. What this project is

NEARLY benchmarks multiple ANN indexing methods for **dense passage retrieval** on **MS MARCO** using **ranking metrics** (NDCG/MRR/MAP) and **system metrics** (latency).

Core idea: ANN is often evaluated by **Recall@K**, but real search systems care about **ranking quality**. This project studies the “Recall Fallacy”:
> higher ANN recall ≠ necessarily higher ranking effectiveness.

---

## 2. Dataset & Task

- **Dataset:** MS MARCO Passage Ranking
- **Corpus size:** 500,000 passages (subset)
- **Queries:** 7,000 dev queries
- **Judged queries (overlap with qrels):** 3,847  
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2` (768-d)
- **Similarity:** Maximum Inner Product Search (MIPS)  
  - implemented as cosine by **L2-normalizing embeddings**, so inner product behaves like cosine similarity.

---

## 3. ANN Algorithms (Target: 8)

### Implemented (Completed)
1. **FlatIP** (Exact baseline on subsets) — FAISS  
2. **HNSW (IP)** — FAISS  
3. **IVF-Flat (IP)** — FAISS  
4. **IVF-PQ (IP)** — FAISS  

### Planned (Pending)
5. **HNSW (IP)** — hnswlib  
6. **HNSW / SW-Graph** — NMSLIB  
7. **Annoy** — tree/forest baseline  
8. **ScaNN (CPU mode)** — hybrid quantization + scoring  

---

## 4. Evaluation Metrics

### Ranking Metrics (User View)
- **NDCG@10**
- **MRR@10**
- **MAP@100**
- **Recall@100**
- **Recall@200**

**Important:** MS MARCO dev has partial judgments.  
Only queries with relevance judgments are evaluated (3,847 out of 7,000).  
If a run retrieves zero relevant docs for all judged queries, some metrics become undefined; we treat those outcomes as **0.0 effectiveness** or explicitly report them as “no relevant retrieved”.

### System Metrics (Infra View)
- Query latency per query (ms)
- p50 / p90 / p95 / p99 latency

---

## 5. Project Layout (What’s in the repo)

