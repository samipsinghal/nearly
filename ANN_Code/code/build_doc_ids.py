import json, os

DOC_JSON = os.path.expanduser("~/documents.json")
OUT_PATH = os.path.expanduser("~/data/msmarco/doc_ids.txt")

def main():
    with open(DOC_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            # documents.json likely uses "id" because queries.json uses "id"
            f.write(str(d["id"]) + "\n")

    print("Wrote:", OUT_PATH)
    print("Count:", len(docs))

if __name__ == "__main__":
    main()

