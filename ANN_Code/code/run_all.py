import os
import time
import glob
import numpy as np
import faiss
import hnswlib
import ngtpy
import nmslib
import scann
from tqdm import tqdm

# --- CONFIG ---
EMB_DIR = "data/embeddings"
INDEX_DIR = "indexes_full"
DIM = 768

os.makedirs(INDEX_DIR, exist_ok=True)

# --- 1. LOAD DATA (RAM MODE) ---
print("🔹 Loading Dataset into RAM...")
files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
if not files: raise FileNotFoundError("No .npy files found!")

# Calculate total size
total_vectors = sum(np.load(f, mmap_mode='r').shape[0] for f in files)
print(f"   Detected {total_vectors:,} vectors.")

# Allocate RAM
data = np.zeros((total_vectors, DIM), dtype='float32')

# Load
idx = 0
for f in tqdm(files, desc="Loading"):
    d = np.load(f)
    c = d.shape[0]
    data[idx : idx + c] = d
    idx += c
print("✅ Data Loaded.")

# ==========================================
# 1. FAISS FLAT (The Oracle)
# ==========================================
p = os.path.join(INDEX_DIR, "faiss_flat.index")
if not os.path.exists(p):
    print("\n[1/8] Building FAISS Flat (Oracle)...")
    t0 = time.time()
    idx = faiss.IndexFlatIP(DIM)
    idx.add(data)
    faiss.write_index(idx, p)
    print(f"✅ Flat Done: {time.time()-t0:.1f}s")

# ==========================================
# 2. FAISS IVF-FLAT
# ==========================================
p = os.path.join(INDEX_DIR, "faiss_ivf_flat.index")
if not os.path.exists(p):
    print("\n[2/8] Building FAISS IVF-Flat...")
    t0 = time.time()
    quantizer = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFFlat(quantizer, DIM, 4096, faiss.METRIC_INNER_PRODUCT)
    idx.verbose = True
    idx.train(data[:1000000]) # Train on 1M
    idx.add(data)
    faiss.write_index(idx, p)
    print(f"✅ IVF-Flat Done: {time.time()-t0:.1f}s")

# ==========================================
# 3. FAISS IVF-PQ (Quantized)
# ==========================================
p = os.path.join(INDEX_DIR, "faiss_ivf_pq.index")
if not os.path.exists(p):
    print("\n[3/8] Building FAISS IVF-PQ...")
    t0 = time.time()
    quantizer = faiss.IndexFlatIP(DIM)
    # M=64 (Sub-vectors), nbits=8
    idx = faiss.IndexIVFPQ(quantizer, DIM, 16384, 64, 8, faiss.METRIC_INNER_PRODUCT)
    idx.verbose = True
    idx.train(data[:1000000])
    idx.add(data)
    faiss.write_index(idx, p)
    print(f"✅ IVF-PQ Done: {time.time()-t0:.1f}s")

# ==========================================
# 4. FAISS HNSW
# ==========================================
p = os.path.join(INDEX_DIR, "faiss_hnsw.index")
if not os.path.exists(p):
    print("\n[4/8] Building FAISS HNSW...")
    t0 = time.time()
    idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 128
    idx.verbose = True
    idx.add(data)
    faiss.write_index(idx, p)
    print(f"✅ FAISS HNSW Done: {time.time()-t0:.1f}s")

# ==========================================
# 5. HNSWLIB (Original)
# ==========================================
p = os.path.join(INDEX_DIR, "hnswlib.bin")
if not os.path.exists(p):
    print("\n[5/8] Building HNSWLIB...")
    t0 = time.time()
    # Normalize for IP (Cosine) simulation
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data_norm = data / (norms + 1e-10)
    
    idx = hnswlib.Index(space='ip', dim=DIM)
    idx.init_index(max_elements=total_vectors, ef_construction=100, M=32)
    idx.add_items(data_norm)
    idx.save_index(p)
    del data_norm # Cleanup
    print(f"✅ HNSWLIB Done: {time.time()-t0:.1f}s")

# ==========================================
# 6. NMSLIB (SW-Graph)
# ==========================================
p = os.path.join(INDEX_DIR, "nmslib.bin")
if not os.path.exists(p):
    print("\n[6/8] Building NMSLIB...")
    t0 = time.time()
    idx = nmslib.init(method='hnsw', space='cosinesimil')
    idx.addDataPointBatch(data)
    idx.createIndex({'M': 32, 'efConstruction': 100}, print_progress=True)
    idx.saveIndex(p, save_data=True)
    print(f"✅ NMSLIB Done: {time.time()-t0:.1f}s")

# ==========================================
# 7. NGT (Yahoo Japan)
# ==========================================
p = os.path.join(INDEX_DIR, "ngt_index")
if not os.path.exists(p):
    print("\n[7/8] Building NGT...")
    t0 = time.time()
    if os.path.exists(p): 
        import shutil
        shutil.rmtree(p)
    ngtpy.create(p, DIM, distance_type="Cosine")
    idx = ngtpy.Index(p)
    idx.batch_insert(data)
    idx.save()
    print(f"✅ NGT Done: {time.time()-t0:.1f}s")

# ==========================================
# 8. Google ScaNN (The Final Boss)
# ==========================================
# ScaNN doesn't save to disk easily like FAISS, so we usually serialize the directory
p = os.path.join(INDEX_DIR, "scann")
if not os.path.exists(p):
    print("\n[8/8] Building ScaNN...")
    t0 = time.time()
    # ScaNN Builder
    searcher = scann.scann_ops_pybind.builder(data, 10, "dot_product") \
        .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000) \
        .score_ah(2, anisotropic_quantization_threshold=0.2) \
        .reorder(100) \
        .build()
    
    os.makedirs(p, exist_ok=True)
    searcher.serialize(p)
    print(f"✅ ScaNN Done: {time.time()-t0:.1f}s")

print("\n🎉 ALL 8 INDEXES BUILT SUCCESSFULLY.")
