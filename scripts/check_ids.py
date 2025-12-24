import json
import sys
import os

def check_json_compatibility(qrels_path, doc_ids_path):
    print(f"--- INSPECTING DATA ---")
    
    # 1. Load Ground Truth (Qrels)
    try:
        with open(qrels_path, 'r') as f:
            qrels_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading Qrels: {e}")
        return

    qrel_doc_ids = set()
    
    # DETECT FORMAT: List vs Dict
    if isinstance(qrels_data, list):
        print("Format detected: LIST (Correct for your pipeline)")
        # Loop through list of dicts: [{'query_id': '...', 'doc_id': '...'}]
        for item in qrels_data:
            if 'doc_id' in item:
                qrel_doc_ids.add(str(item['doc_id']).strip())
    elif isinstance(qrels_data, dict):
        print("Format detected: DICT")
        for qid, docs in qrels_data.items():
            if isinstance(docs, dict):
                for doc_id in docs.keys():
                    qrel_doc_ids.add(str(doc_id).strip())
            elif isinstance(docs, list):
                for doc_id in docs:
                    qrel_doc_ids.add(str(doc_id).strip())

    if not qrel_doc_ids:
        print("❌ CRITICAL: Could not find any 'doc_id' fields in Qrels.")
        return

    print(f"Unique Target Doc IDs in QRELS: {len(qrel_doc_ids)}")
    print(f"Sample QREL ID: '{list(qrel_doc_ids)[0]}'")

    # 2. Load Database IDs (Doc IDs)
    try:
        with open(doc_ids_path, 'r') as f:
            doc_pool = json.load(f)
    except Exception as e:
        print(f"❌ Error loading Doc IDs: {e}")
        return
        
    doc_pool_set = set()
    
    if isinstance(doc_pool, list):
        doc_pool_set = set(str(x).strip() for x in doc_pool)
    elif isinstance(doc_pool, dict):
        # If it's a mapping like {"id": index}, take keys
        doc_pool_set = set(str(k).strip() for k in doc_pool.keys())
        
    print(f"Total Database IDs: {len(doc_pool_set)}")
    if len(doc_pool_set) > 0:
        print(f"Sample Database ID: '{list(doc_pool_set)[0]}'")

    # 3. Check Overlap
    missing = qrel_doc_ids - doc_pool_set
    match_rate = 100 - (len(missing) / len(qrel_doc_ids) * 100) if qrel_doc_ids else 0
    
    print(f"\n--- COMPATIBILITY RESULT ---")
    print(f"Overlap: {match_rate:.2f}%")
    
    if match_rate < 100:
        print(f"⚠️ WARNING: {len(missing)} IDs in Qrels are NOT found in your Doc IDs list.")
        print(f"Example Missing IDs: {list(missing)[:3]}")
    else:
        print("✅ PASSED: All Qrel IDs exist in your Doc ID list.")

if __name__ == "__main__":
    # Pointing to the specific files you identified earlier
    qrels = "data/raw/qrels.json"
    docs = "data/embeddings/doc_ids.json"
    
    if os.path.exists(qrels) and os.path.exists(docs):
        check_json_compatibility(qrels, docs)
    else:
        print(f"Files not found.\nChecking: {qrels}\nChecking: {docs}")