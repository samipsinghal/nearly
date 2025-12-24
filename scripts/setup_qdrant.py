from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

# Connect to local Qdrant
client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "nearly_bench"

def initialize_db():
    print(f"Creating collection: {COLLECTION_NAME}")
    
    # recreate_collection will wipe it clean if it already exists
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,               # mpnet-base-v2
            distance=Distance.COSINE, 
            on_disk=True            # Keeps vectors on SSD
        ),
        hnsw_config=HnswConfigDiff(
            on_disk=True,           # Keeps search index on SSD
            m=16,                   # Standard complexity for HNSW
            ef_construct=100        # Quality of index build
        )
    )
    print("✅ Collection ready.")

if __name__ == "__main__":
    initialize_db()