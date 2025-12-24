import os
import time
import numpy as np
import faiss

# --- Try imports ---
try: import hnswlib
except: hnswlib = None
try: import ngtpy
except: ngtpy = None
try: import nmslib
except: nmslib = None

# --- Config ---
MMAP_FILE = "data/corpus_full_mmap.dat"
INDEX_DIR = "indexes_full"
DIM = 768
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Load Data Virtualization ---
if not os.path.exists(MMAP_FILE):
    print("❌ Run 03a_prepare_full_mmap.py first!")
    exit()

file_size = os.path.getsize(MMAP_FILE)
num_vectors = file_size // (DIM * 4)
data_mmap = np.memmap(MMAP_FILE, dtype='float32', mode='r', shape=(num_vectors, DIM))

print(f"Dataset: {num_vectors:,} vectors loaded virtually.")

def train_sample(size=250_000):
    indices = np.random.choice(num_vectors, size=size, replace=False)
    return data_mmap[indices].copy()

# 1. FAISS IVF-FLAT (Baseline)
path_ivf = os.path.join(INDEX_DIR, "faiss_ivf_flat.index")
if not os.path.exists(path_ivf):
    print("\n[1/7] Building FAISS IVF-Flat...")
    t0 = time.time()
    train_data = train_sample()
    faiss.normalize_L2(train_data)
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, 4096, faiss.METRIC_INNER_PRODUCT)
    index.train(train_data)
    del train_data
    
    batch_size = 500_000
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        batch = data_mmap[i:end].copy()
        faiss.normalize_L2(batch)
        index.add(batch)
        del batch
    faiss.write_index(index, path_ivf)
    print(f"✅ IVF-Flat Done ({time.time()-t0:.1f}s)")

# 2. FAISS IVF-PQ (Compression)
path_pq = os.path.join(INDEX_DIR, "faiss_ivf_pq.index")
if not os.path.exists(path_pq):
    print("\n[2/7] Building FAISS IVF-PQ...")
    t0 = time.time()
    train_data = train_sample()
    faiss.normalize_L2(train_data)
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFPQ(quantizer, DIM, 16384, 64, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(train_data)
    del train_data
    
    batch_size = 500_000
    for i in range(0, num_vectors, batch_size):
        end = min(i + batch_size, num_vectors)
        batch = data_mmap[i:end].copy()
        faiss.normalize_L2(batch)
        index.add(batch)
        del batch
    faiss.write_index(index, path_pq)
    print(f"✅ IVF-PQ Done ({time.time()-t0:.1f}s)")

# 3. FAISS HNSW (Graph)
path_hnsw = os.path.join(INDEX_DIR, "faiss_hnsw.index")
if not os.path.exists(path_hnsw):
    print("\n[3/7] Building FAISS HNSW (High RAM)...")
    try:
        t0 = time.time()
        index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 128
        batch_size = 200_000
        for i in range(0, num_vectors, batch_size):
            end = min(i + batch_size, num_vectors)
            batch = data_mmap[i:end].copy()
            faiss.normalize_L2(batch)
            index.add(batch)
            del batch
        faiss.write_index(index, path_hnsw)
        print(f"✅ FAISS HNSW Done ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"❌ FAISS HNSW Failed: {e}")

# 4. HNSWLIB
if hnswlib:
    path_lib = os.path.join(INDEX_DIR, "hnswlib.bin")
    if not os.path.exists(path_lib):
        print("\n[4/7] Building HNSWLIB...")
        try:
            t0 = time.time()
            p = hnswlib.Index(space='ip', dim=DIM)
            p.init_index(max_elements=num_vectors, ef_construction=100, M=16)
            batch_size = 500_000
            for i in range(0, num_vectors, batch_size):
                end = min(i + batch_size, num_vectors)
                batch = data_mmap[i:end].copy()
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                batch = batch / (norms + 1e-10)
                p.add_items(batch)
                del batch
            p.save_index(path_lib)
            print(f"✅ HNSWLIB Done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"❌ HNSWLIB Failed: {e}")

# 5. NGT
if ngtpy:
    path_ngt = os.path.join(INDEX_DIR, "ngt_index")
    if not os.path.exists(path_ngt):
        print("\n[5/7] Building NGT...")
        try:
            t0 = time.time()
            ngtpy.create(path_ngt, DIM, distance_type="Cosine")
            index = ngtpy.Index(path_ngt)
            batch_size = 500_000
            for i in range(0, num_vectors, batch_size):
                end = min(i + batch_size, num_vectors)
                print(f"    Batch {i}-{end}...")
                batch = data_mmap[i:end].copy()
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                batch = batch / (norms + 1e-10)
                index.batch_insert(batch)
                del batch
            index.save()
            print(f"✅ NGT Done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"❌ NGT Failed: {e}")

# 6. NMSLIB
if nmslib:
    path_nms = os.path.join(INDEX_DIR, "nmslib_sw.bin")
    if not os.path.exists(path_nms):
        print("\n[6/7] Building NMSLIB (SW-Graph)...")
        try:
            t0 = time.time()
            index = nmslib.init(method='hnsw', space='cosinesimil')
            print("    Feeding data to NMSLIB...")
            index.addDataPointBatch(data_mmap)
            index.createIndex({'M': 32, 'efConstruction': 100}, print_progress=True)
            index.saveIndex(path_nms, save_data=True)
            print(f"✅ NMSLIB Done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"❌ NMSLIB Failed: {e}")

print("\n🎉 Indexing Process Complete.")