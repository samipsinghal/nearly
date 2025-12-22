"""
05_evaluate.py

Evaluates TREC run files against MS MARCO qrels using ir_measures.

Inputs:
- data/raw/qrels.json
- runs/*.run

Outputs:
- metrics/eval_results.json
- metrics/eval_results.csv
"""

import os
import json
import glob
import csv

from typing import Dict, List, Tuple

RUNS_DIR = "runs"
RAW_DIR = "data/raw"
OUT_DIR = "metrics"
os.makedirs(OUT_DIR, exist_ok=True)

QRELS_JSON = os.path.join(RAW_DIR, "qrels.json")

# Primary ranking metrics
# Note: MS MARCO often reports MRR@10.
MEASURES = [
    "nDCG@10",
    "MRR@10",
    "MAP@100",
    "Recall@100",
    "Recall@200",
]

def load_qrels_trec(qrels_json_path: str):
    """
    Returns:
      - qrels_str (TREC format)
      - judged_qids (set of query IDs with at least one judgment)
    """
    with open(qrels_json_path, "r") as f:
        qrels = json.load(f)

    lines = []
    judged_qids = set()

    for r in qrels:
        qid = r["query_id"]
        docid = r["doc_id"]
        rel = int(r["relevance"])
        judged_qids.add(qid)
        lines.append(f"{qid} 0 {docid} {rel}")

    return "\n".join(lines) + "\n", judged_qids


def main():
    from ir_measures import calc_aggregate, read_trec_run, read_trec_qrels
    from ir_measures import nDCG, MRR, MAP, Recall

    # Build measure objects
    measure_objs = []
    for m in MEASURES:
        name, k = m.split("@")
        k = int(k)
        if name == "nDCG":
            measure_objs.append(nDCG@k)
        elif name == "MRR":
            measure_objs.append(MRR@k)
        elif name == "MAP":
            measure_objs.append(MAP@k)
        elif name == "Recall":
            measure_objs.append(Recall@k)
        else:
            raise ValueError(f"Unknown measure: {m}")

    print("Loading qrels...")
    qrels_str, judged_qids = load_qrels_trec(QRELS_JSON)
    qrels = read_trec_qrels(qrels_str)

    run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "*.run")))
    if not run_files:
        raise RuntimeError("No run files found in runs/")

    results: List[Dict] = []

    print(f"Evaluating {len(run_files)} run files...")
    for rf in run_files:
        run_name = os.path.basename(rf).replace(".run", "")
        print(" -", run_name)

        run = read_trec_run(rf)
        run = (
        r for r in run
        if r.query_id in judged_qids
        )


        agg = calc_aggregate(measure_objs, qrels, run)

        row = {"run": run_name}
        # agg keys are measure objects, convert to friendly names
        for meas, val in agg.items():
            row[str(meas)] = float(val)
        results.append(row)

    # Save JSON
    out_json = os.path.join(OUT_DIR, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "eval_results.csv")
    # Collect union of all keys
    keys = ["run"]
    for r in results:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("Saved:", out_json)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
