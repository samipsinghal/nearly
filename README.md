# NEARLY: NEAR-neighbor Retrieval anaLYsis
### A ranking-centric benchmark of ANN methods for dense retrieval on MS MARCO

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Library](https://img.shields.io/badge/library-FAISS%20%7C%20HNSWlib-orange)
![Infrastructure](https://img.shields.io/badge/hardware-GPU%20%2B%20CPU-blueviolet)

---

## 1. 🎯 What this project is

**NEARLY** benchmarks multiple Approximate Nearest Neighbor (ANN) indexing methods for **dense passage retrieval** on the **MS MARCO** dataset. Unlike standard ANN benchmarks that focus solely on recall, this project prioritizes **ranking metrics** (NDCG, MRR, MAP) alongside **system metrics** (latency).

### The Core Problem
ANN is traditionally evaluated by **Recall@K**. However, real search systems care about the final ranking quality presented to the user. This project investigates the "Recall Fallacy":

> **Hypothesis:** Higher ANN Recall@K does not necessarily equal higher Ranking Effectiveness (NDCG/MRR).

---

## 2. 📊 Dataset & Task

* **Dataset:** MS MARCO Passage Ranking (subset)
* **Corpus Size:** 500,000 passages
* **Queries:** 7,000 dev queries
* **Judged Queries:** 3,847 (subset with relevance judgments used for evaluation)
* **Embedding Model:** `sentence-transformers/all-mpnet-base-v2` (768-d)
* **Similarity Metric:** Maximum Inner Product Search (MIPS)
    * *Note: Implemented as cosine similarity via L2-normalization of embeddings.*

---

## 3. 🤖 ANN Algorithms (Target: 8)

### ✅ Implemented
1.  **FlatIP (Exact Baseline)** — FAISS (Brute force)
2.  **HNSW (IP)** — FAISS
3.  **IVF-Flat (IP)** — FAISS
4.  **IVF-PQ (IP)** — FAISS

### 🚧 Planned / In Progress
5.  **HNSW (IP)** — hnswlib
6.  **HNSW / SW-Graph** — NMSLIB
7.  **Annoy** — Spotify (Tree/Forest baseline)
8.  **ScaNN** — Google (Quantization + Scoring)

---

## 4. 📉 Evaluation Metrics

### Ranking Metrics (User Quality)
* **NDCG@10:** Normalized Discounted Cumulative Gain
* **MRR@10:** Mean Reciprocal Rank
* **MAP@100:** Mean Average Precision
* **Recall@100 / @200**

> **Note on Judgments:** MS MARCO dev judgments are sparse. Only the 3,847 queries with known relevant documents are included in the final metric calculation. Zero retrieval is treated as **0.0 effectiveness**.

### System Metrics (Infrastructure Cost)
* **Latency:** Per-query retrieval time (ms)
* **Percentiles:** p50, p90, p95, p99

---

## 5. ☁️ Infrastructure Strategy

To ensure reproducible and fair latency comparisons, this benchmark adopts a strict hardware strategy:

* **Environment:** [e.g., Google Colab Pro / AWS g4dn.xlarge]
* **GPU:** NVIDIA [e.g., T4 / V100] (Used for heavy indexing)
* **CPU:** [e.g., Intel Xeon, 4 vCPUs] (Used for realistic latency simulation)

**Execution Strategy:**
1.  **Index:** Build on GPU (where supported) for speed.
2.  **Search:** Query on CPU to simulate standard production retrieval nodes.

---

## 6. 📜 Cloud Automation & Scripts

To facilitate running this benchmark on remote cloud VMs (headless), the `scripts/` directory contains automation tools:

* **`setup_vm.sh`**: One-click environment setup.
    * Installs system dependencies (CUDA, C++ build tools).
    * Installs Python requirements (`faiss-gpu`, `sentence-transformers`).
    * Downloads and extracts the MS MARCO dataset and pre-computed embeddings.

* **`run_batch.sh`**:
    * Executes the full benchmark suite in the background (using `nohup`).
    * Ensures experiments continue running even if the SSH session disconnects.
    * Logs all `stdout`/`stderr` to `results/logs/`.

---

## 7. 📂 Project Layout

```text
.
├── data/                   # Dataset storage
│   ├── corpus/             # Raw MS MARCO passages
│   └── embeddings/         # Pre-computed .npy embedding files
├── scripts/                # Cloud automation
│   ├── setup_vm.sh         # Installs deps & downloads data
│   └── run_batch.sh        # Runs experiments in background
├── src/                    # Source code
│   ├── algorithms/         # Wrappers for FAISS, HNSWlib, etc.
│   ├── evaluation/         # Scoring scripts (NDCG, MRR calc)
│   └── utils/              # Data loaders and preprocessing
├── results/                # Output logs, CSV reports, and plots
├── main.py                 # Entry point to run the benchmark
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation