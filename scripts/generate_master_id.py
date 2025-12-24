import json
import os

# CONFIG
# Make sure this points to your big 8.8M line file
RAW_DATA_PATH = "data_raw/documents.json" 
OUTPUT_PATH = "indexes/master_doc_ids.json"

def generate_ids():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: Could not find {RAW_DATA_PATH}")
        return

    print(f"Reading IDs from {RAW_DATA_PATH}...")
    
    doc_ids = []
    count = 0
    
    # Read the file line by line to extract just the IDs
    # This prevents loading the whole 8GB text file into RAM
    with open(RAW_DATA_PATH, 'r') as f:
        for line in f:
            try:
                # Try JSON format first: {"doc_id": "...", "text": "..."}
                data = json.loads(line)
                doc_ids.append(str(data['doc_id'])) 
            except:
                # Fallback for TSV: "1234\tThis is text..."
                parts = line.strip().split('\t')
                if parts:
                    doc_ids.append(parts[0])
            
            count += 1
            if count % 1000000 == 0:
                print(f"  Processed {count} lines...")

    print(f"✅ Extracted {len(doc_ids)} IDs.")
    
    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(doc_ids, f)
    print("Done.")

if __name__ == "__main__":
    generate_ids()