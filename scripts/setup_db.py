from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

# Connect to the Docker container
client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "nearly_bench"

def initialize_db():
    print(f" Initializing collection: {COLLECTION_NAME}")
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,               # all-mpnet-base-v2
            distance=Distance.COSINE, 
            on_disk=True            # Keep vectors on SSD to save RAM
        ),
        hnsw_config=HnswConfigDiff(
            on_disk=True,           # Keep search graph on SSD
            m=16,                   # Balanced graph complexity
            ef_construct=100        # Quality of indexing
        )
    )
    print(" Collection created successfully!")

if __name__ == "__main__":
    initialize_db()