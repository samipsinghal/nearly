import os
import json
import ir_datasets
from google.cloud import storage
from tqdm import tqdm

# --- CONFIG ---
BUCKET_NAME = "nearly-search-data"  # <--- CHANGE THIS IF NEEDED
LOCAL_DIR = "data_raw"
os.makedirs(LOCAL_DIR, exist_ok=True)

def upload_blob(source, dest):
    print(f"Uploading {source} -> gs://{BUCKET_NAME}/{dest}...")
    storage.Client().bucket(BUCKET_NAME).blob(dest).upload_from_filename(source)

def main():
    print("--- 1. PREPARING DATA ON M4 MAX ---")
    
    # 1. Download MS MARCO (8.8M Docs)
    docs_path = os.path.join(LOCAL_DIR, "documents.json")
    if not os.path.exists(docs_path):
        print("Streaming docs from ir_datasets (this may take a while)...")
        dataset = ir_datasets.load("msmarco-passage")
        with open(docs_path, "w") as f:
            f.write("[\n")
            for i, doc in enumerate(tqdm(dataset.docs_iter())):
                if i > 0: f.write(",\n")
                f.write(json.dumps({"id": doc.doc_id, "text": doc.text}))
            f.write("\n]")
    
    upload_blob(docs_path, "raw/documents.json")

    # 2. Prepare Queries (Test questions)
    queries_path = os.path.join(LOCAL_DIR, "queries.json")
    print("Saving queries...")
    dev_set = ir_datasets.load("msmarco-passage/dev/small")
    queries = [{"id": q.query_id, "text": q.text} for q in dev_set.queries_iter()]
    with open(queries_path, "w") as f:
        json.dump(queries, f)
        
    upload_blob(queries_path, "raw/queries.json")
    print("✅ Upload Complete. Ready for Cloud GPU.")

if __name__ == "__main__":
    main()