import time
import psutil
import os
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class NEARLYBenchmark:
    def __init__(self, collection_name="nearly_bench"):
        self.client = QdrantClient("http://localhost:6333")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.collection = collection_name
        self.process = psutil.Process(os.getpid())

    def capture_metrics(self, func, *args, **kwargs):
        """Wrapper to measure resource usage during execution."""
        # Baseline before
        start_mem = self.process.memory_info().rss / (1024 * 1024) # MB
        self.process.cpu_percent(None) # Reset CPU tracker
        
        start_time = time.perf_counter()
        
        # Execute Search
        results = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        
        # Capture After
        end_mem = self.process.memory_info().rss / (1024 * 1024)
        cpu_usage = self.process.cpu_percent(None)
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "results": results,
            "latency_ms": round(latency_ms, 2),
            "memory_delta_mb": round(end_mem - start_mem, 2),
            "memory_total_mb": round(end_mem, 2),
            "cpu_percent": cpu_usage
        }

    def search_task(self, query_text):
        query_vector = self.model.encode(query_text).tolist()
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=10
        )

if __name__ == "__main__":
    bench = NEARLYBenchmark()
    
    queries = ["What is the capital of France?", "Symptoms of a cold"]
    
    print(f"{'Query':<30} | {'Latency':<10} | {'CPU%':<6} | {'RAM (MB)':<10}")
    print("-" * 65)
    
    for q in queries:
        metrics = bench.capture_metrics(bench.search_task, q)
        print(f"{q[:30]:<30} | {metrics['latency_ms']:>7}ms | {metrics['cpu_percent']:>5}% | {metrics['memory_total_mb']:>8}MB")
