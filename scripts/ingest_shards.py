import os
import ijson
import logging
import numpy as np
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# --- Configuration ---
BUCKET_NAME = "nearly-search-data"
COLLECTION_NAME = "nearly_bench"
SHARD_SIZE = 200_000
TOTAL_SHARDS = 45
LOCAL_DOCS_PATH = "data/documents.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_ingestion():
    # 1. Initialize Clients
    client = QdrantClient("http://localhost:6333")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # 2. Check Resumability
    try:
        res = client.get_collection(COLLECTION_NAME)
        current_count = res.points_count
    except Exception:
        current_count = 0
    
    start_shard = current_count // SHARD_SIZE
    logger.info(f"Database contains {current_count} points. Resuming from Shard {start_shard}.")

    # 3. Stream Metadata efficiently
    with open(LOCAL_DOCS_PATH, "rb") as f:
        # ijson.items reads the array one-by-one to save your Mac's RAM
        metadata_stream = ijson.items(f, 'item')
        
        # Fast-forward to where we left off
        if current_count > 0:
            logger.info(f"Fast-forwarding stream past {current_count} records...")
            for _ in range(current_count):
                next(metadata_stream, None)

        # 4. Ingestion Loop
        for shard_idx in range(start_shard, TOTAL_SHARDS):
            blob_name = f"embeddings/doc_emb_{shard_idx:03d}.npy"
            temp_file = f"temp_shard_{shard_idx}.npy"
            
            try:
                logger.info(f"--- Processing Shard {shard_idx}/{TOTAL_SHARDS-1} ---")
                bucket.blob(blob_name).download_to_filename(temp_file)
                vectors = np.load(temp_file)
                
                points = []
                for vec in vectors:
                    meta = next(metadata_stream, None)
                    if meta is None: break
                    
                    # Safely find the ID (MS MARCO uses 'id', 'doc_id', or 'pid')
                    point_id = meta.get('id') or meta.get('doc_id') or meta.get('pid')
                    
                    points.append(PointStruct(
                        id=int(point_id),
                        vector=vec.tolist(),
                        payload=meta
                    ))

                    # Batch upsert to Qdrant (1000 is the sweet spot for throughput)
                    if len(points) >= 1000:
                        client.upsert(COLLECTION_NAME, points=points, wait=False)
                        points = []

                # Final flush for the shard
                if points:
                    client.upsert(COLLECTION_NAME, points=points, wait=True)
                
                logger.info(f"✅ Successfully indexed Shard {shard_idx}.")
                
            except Exception as e:
                logger.error(f"Failed shard {shard_idx}: {e}")
                continue
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    logger.info("🎉 Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()