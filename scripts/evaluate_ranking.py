import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("http://localhost:6333")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def benchmark_query(query_text, limit=10):
    start_time = time.time()
    
    # 1. Embed the query
    query_vector = model.encode(query_text).tolist()
    
    # 2. Search in Qdrant
    results = client.search(
        collection_name="nearly_bench",
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    
    latency = (time.time() - start_time) * 1000 # in ms
    return results, latency

if __name__ == "__main__":
    # Example Test
    test_query = "how to setup a vector database"
    hits, ms = benchmark_query(test_query)
    
    print(f"Search took {ms:.2f}ms")
    for i, hit in enumerate(hits):
        print(f"{i+1}. [Score: {hit.score:.4f}] {hit.payload['text'][:80]}...")