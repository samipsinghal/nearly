import os
import time
import glob
import numpy as np
import faiss
import hnswlib
import ngtpy
import nmslib
from tqdm import tqdm

# --- CONFIG ---
EMB_DIR = "data/embeddings"
MMAP_FILE = "data/corpus_full_mmap.dat"
INDEX_DIR = "indexes_full"
DIM = 768

# --- 1. LOAD DATA INTO PURE RAM ---
print("🔹 Loading Dataset...")

# Get list of files
files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
if not files: raise FileNotFoundError("No .npy files found in data/embeddings!")

# Calculate total size first
total_vectors = sum(np.load(f, mmap_mode='r').shape[0] for f in files)
print(f"   Detected {total_vectors:,} vectors.")

# Allocate RAM (This will take ~26GB RAM, which is easy for your 251GB machine)
data = np.zeros((total_vectors, DIM), dtype='float32')

# Load data into RAM
idx = 0
for f in tqdm(files, desc="Loading NPYs"):
    d = np.load(f)
    c = d.shape[0]
    data[idx : idx + c] = d
    idx += c

print(f"✅ Loaded {data.shape[0]:,} vectors into RAM.")


# --- 2. BUILD INDEXES ---

# A. FAISS IVF-FLAT (Baseline)
p = os.path.join(INDEX_DIR, "ivf_flat.index")
if not os.path.exists(p):
    print("\n🚀 [1/4] Building FAISS IVF-Flat...")
    t0 = time.time()
    
    # Train on a massive 1M vector sample for high accuracy
    quantizer = faiss.IndexFlatIP(DIM)
    idx = faiss.IndexIVFFlat(quantizer, DIM, 4096, faiss.METRIC_INNER_PRODUCT)
    idx.verbose = True
    
    print("   Training...")
    idx.train(data[:1000000]) # Use first 1M for training
    
    print("   Indexing...")
    idx.add(data) # Add all at once
    
    faiss.write_index(idx, p)
    print(f"✅ IVF-Flat Done: {time.time()-t0:.1f}s")

# B. FAISS HNSW (The Standard)
p = os.path.join(INDEX_DIR, "hnsw_faiss.index")
if not os.path.exists(p):
    print("\n🚀 [2/4] Building FAISS HNSW...")
    t0 = time.time()
    
    idx = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 128
    idx.verbose = True
    
    print("   Indexing (This is CPU heavy)...")
    idx.add(data) # Add all at once
    
    faiss.write_index(idx, p)
    print(f"✅ FAISS HNSW Done: {time.time()-t0:.1f}s")

# C. NMSLIB (The Speed Demon)
p = os.path.join(INDEX_DIR, "nmslib.bin")
if not os.path.exists(p):
    print("\n🚀 [3/4] Building NMSLIB (SW-Graph)...")
    t0 = time.time()
    
    # Init NMSLIB
    idx = nmslib.init(method='hnsw', space='cosinesimil')
    
    print("   Feeding data...")
    idx.addDataPointBatch(data) 
    
    print("   Building Graph...")
    idx.createIndex({'M': 32, 'efConstruction': 100}, print_progress=True)
    
    idx.saveIndex(p, save_data=True)
    print(f"✅ NMSLIB Done: {time.time()-t0:.1f}s")

# D. NGT (Yahoo Japan - Often Fastest)
p = os.path.join(INDEX_DIR, "ngt_index")
if not os.path.exists(p):
    print("\n🚀 [4/4] Building NGT...")
    t0 = time.time()
    if os.path.exists(p): 
        import shutil
        shutil.rmtree(p)
        
    ngtpy.create(p, DIM, distance_type="Cosine")
    idx = ngtpy.Index(p)
    
    print("   Batch Inserting...")
    idx.batch_insert(data) 
    
    idx.save()
    print(f"✅ NGT Done: {time.time()-t0:.1f}s")

print("\n🎉 BENCHMARK COMPLETE.")
