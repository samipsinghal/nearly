import time
import numpy as np
import faiss
import hnswlib
import nmslib
import annoy
import scann

class BaseANN:
    """Standard interface so we don't have to rewrite the benchmark loop every time."""
    def __init__(self, name):
        self.name = name
        self.build_time = 0
    
    def build(self, data):
        raise NotImplementedError
    
    def search(self, queries, k):
        raise NotImplementedError

# --- FAISS Implementations ---

class FaissFlat(BaseANN):
    def __init__(self):
        super().__init__("FAISS_FlatIP_Exact")

    def build(self, data):
        # Using FlatIP (Inner Product) because our vectors are normalized.
        # This is effectively Cosine Similarity but faster.
        print(f"[{self.name}] Indexing {data.shape[0]} vectors...")
        start = time.time()
        self.index = faiss.IndexFlatIP(data.shape[1])
        self.index.add(data)
        self.build_time = time.time() - start

    def search(self, queries, k):
        start = time.time()
        D, I = self.index.search(queries, k)
        return I, D, time.time() - start

class FaissHNSW(BaseANN):
    def __init__(self, M=32, efConstruction=200):
        super().__init__(f"FAISS_HNSW_M{M}")
        self.M = M
        self.efConstruction = efConstruction

    def build(self, data):
        print(f"[{self.name}] Building graph...")
        start = time.time()
        # METRIC_INNER_PRODUCT is critical here. If you use L2, you break the ranking.
        self.index = faiss.IndexHNSWFlat(data.shape[1], self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.efConstruction
        self.index.add(data)
        self.build_time = time.time() - start

    def search(self, queries, k):
        # efSearch is a runtime trade-off. 128 is a safe default for high recall.
        # Lower this if you want to cheat on latency.
        self.index.hnsw.efSearch = 128
        start = time.time()
        D, I = self.index.search(queries, k)
        return I, D, time.time() - start

class FaissIVFPQ(BaseANN):
    def __init__(self, nlist=1024, m=8):
        super().__init__(f"FAISS_IVF_PQ_m{m}")
        self.nlist = nlist
        self.m = m 

    def build(self, data):
        d = data.shape[1]
        print(f"[{self.name}] Training Quantizer (IVF)...")
        start = time.time()
        
        # IVFPQ needs a quantizer (coarse) and sub-quantizers (fine).
        # We use FlatIP for the coarse quantizer.
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8, faiss.METRIC_INNER_PRODUCT)
        
        # IVF requires training on the vector distribution first.
        self.index.train(data)
        self.index.add(data)
        self.build_time = time.time() - start

    def search(self, queries, k):
        self.index.nprobe = 16 # Check top 16 clusters. 
        start = time.time()
        D, I = self.index.search(queries, k)
        return I, D, time.time() - start

# --- Other Libraries ---

class HnswlibIndex(BaseANN):
    def __init__(self, M=32):
        super().__init__(f"Hnswlib_M{M}")
        self.M = M

    def build(self, data):
        print(f"[{self.name}] Init index...")
        start = time.time()
        # 'ip' = Inner Product. 
        self.index = hnswlib.Index(space='ip', dim=data.shape[1])
        
        # Declaring max_elements is mandatory in hnswlib (unlike FAISS).
        # If we exceed this, it crashes.
        self.index.init_index(max_elements=data.shape[0], ef_construction=200, M=self.M)
        self.index.add_items(data)
        self.build_time = time.time() - start

    def search(self, queries, k):
        self.index.set_ef(128)
        start = time.time()
        I, D = self.index.knn_query(queries, k=k)
        return I, D, time.time() - start

class NmslibIndex(BaseANN):
    def __init__(self):
        super().__init__("NMSLIB_HNSW")

    def build(self, data):
        print(f"[{self.name}] Adding data...")
        # 'cosinesimil' is NMSLIB's way of handling normalized dot product
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        
        start = time.time()
        self.index.addDataPointBatch(data)
        # NMSLIB builds index *after* adding data.
        self.index.createIndex({'M': 32, 'efConstruction': 200}, print_progress=True)
        self.build_time = time.time() - start

    def search(self, queries, k):
        self.index.setQueryTimeParams({'efSearch': 128})
        start = time.time()
        results = self.index.knnQueryBatch(queries, k=k, num_threads=4)
        
        # NMSLIB returns a list of tuples, annoying format to unpack
        I = np.array([r[0] for r in results])
        D = np.array([r[1] for r in results])
        return I, D, time.time() - start

class AnnoyIndex(BaseANN):
    def __init__(self, n_trees=100):
        super().__init__(f"Annoy_Trees{n_trees}")
        self.n_trees = n_trees

    def build(self, data):
        print(f"[{self.name}] Building trees...")
        start = time.time()
        # Annoy is older. 'dot' is safe here.
        self.index = annoy.AnnoyIndex(data.shape[1], 'dot')
        
        # Annoy is slow to add items one-by-one in Python loop, but
        # it's the only way (no batch add).
        for i, vector in enumerate(data):
            self.index.add_item(i, vector)
            
        self.index.build(self.n_trees)
        self.build_time = time.time() - start

    def search(self, queries, k):
        start = time.time()
        I, D = [], []
        # No batch search in Annoy, so we loop. It kills QPS.
        for q in queries:
            ids, dists = self.index.get_nns_by_vector(q, k, search_k=-1, include_distances=True)
            I.append(ids)
            D.append(dists)
        return np.array(I), np.array(D), time.time() - start

class ScannIndex(BaseANN):
    def __init__(self):
        super().__init__("Google_ScaNN")

    def build(self, data):
        print(f"[{self.name}] Training ScaNN (this includes reordering)...")
        start = time.time()
        # This config string is sensitive.
        # num_leaves=2000 is rule-of-thumb for 500k-1M datasets.
        # anisotropic_quantization_threshold=0.2 is the magic number from the paper.
        self.searcher = scann.scann_ops_pybind.builder(data, 10, "dot_product") \
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=100000) \
            .score_ah(2, anisotropic_quantization_threshold=0.2) \
            .reorder(100) \
            .build()
        self.build_time = time.time() - start

    def search(self, queries, k):
        start = time.time()
        # Modern ScaNN supports batching, but fallback to loop if version mismatch
        try:
            I, D = self.searcher.search_batched(queries, final_num_neighbors=k)
        except AttributeError:
            I, D = [], []
            for q in queries:
                ids, dists = self.searcher.search(q, final_num_neighbors=k)
                I.append(ids)
                D.append(dists)
            I, D = np.array(I), np.array(D)
            
        return I, D, time.time() - start