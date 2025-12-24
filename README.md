# NEARLY: NEAR-neighbor Retrieval anaLYsis

## A ranking-centric benchmark of ANN methods for dense retrieval on MS MARCO



### 1. What this project is

NEARLY is a benchmarking framework for Approximate Nearest Neighbor (ANN) methods applied to dense passage retrieval on the MS MARCO dataset.
Unlike traditional ANN benchmarks that focus primarily on Recall@K, this project emphasizes ranking effectiveness—the quality of the final ranked list presented to users—alongside system performance metrics such as latency.
The Core Problem
ANN systems are often evaluated using Recall@K, which measures whether relevant items appear anywhere in the candidate set. However, real-world retrieval systems care about where relevant documents appear in the ranking.
This project investigates the Recall Fallacy:
Hypothesis: Higher ANN Recall@K does not necessarily imply higher ranking effectiveness (e.g., NDCG or MRR).
NEARLY explicitly measures how ANN design choices affect end-to-end ranking quality, not just candidate recall.

### 2. Dataset & Retrieval Task
Dataset: MS MARCO Passage Ranking
Corpus Size: ~500,000 passages (subset used for experimentation)
Queries: ~7,000 development queries
Judged Queries: 3,847 queries with relevance judgments
Embedding Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
Similarity Metric: Maximum Inner Product Search (MIPS)
Implemented via cosine similarity using L2-normalized embeddings
Only queries with relevance judgments are included in metric computation. Queries with no retrieved relevant documents contribute zero effectiveness, consistent with IR evaluation practice.

### 3. ANN Algorithms
Target: 8 ANN methods

Implemented
FlatIP (Exact Baseline) — FAISS brute-force search
HNSW (IP) — FAISS
IVF-Flat (IP) — FAISS
IVF-PQ (IP) — FAISS
HNSW (IP) — hnswlib
HNSW / SW-Graph — NMSLIB
Annoy — Spotify (tree-based baseline)
ScaNN — Google (quantization + scoring)

### 4. Evaluation Metrics
Ranking Metrics (User-Facing Quality)
NDCG@10 — Normalized Discounted Cumulative Gain
MRR@10 — Mean Reciprocal Rank
MAP@100 — Mean Average Precision
Recall@100 / Recall@200
Note: MS MARCO relevance judgments are sparse. Evaluation is restricted to judged queries, and failure to retrieve a relevant document yields a score of 0.0.
System Metrics (Efficiency & Cost)
Latency: Per-query retrieval time (milliseconds)
Latency Percentiles: p50, p90, p95, p99
Throughput: Queries per second (QPS)

### 5. Infrastructure Strategy
To ensure fair and reproducible latency measurements, NEARLY uses a strict execution strategy:
Environment: Cloud VM (GCP)
GPU: NVIDIA L4 (used for embedding and index construction)
CPU: Multi-core x86 CPU (used for search evaluation)
Execution Model
Index Construction: Performed on GPU (where supported) for scalability.
Query Evaluation: Executed on CPU to simulate production retrieval nodes.
This separation prevents GPU acceleration from artificially inflating search-time performance.
