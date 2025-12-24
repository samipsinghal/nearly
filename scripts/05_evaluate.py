"""
05_evaluate.py

Evaluates TREC run files against the local QRELS file generated in Step 1.
Forces all IDs to strings to prevent "NaN" errors due to type mismatches.
"""

import os
import glob
import json
import pandas as pd
import ir_measures
from ir_measures import nDCG, MRR, Recall

# --- Config ---
RUNS_DIR = "runs"
RESULTS_DIR = "results"
QRELS_PATH = "data/raw/qrels.json"  # Using your local file
os.makedirs(RESULTS_DIR, exist_ok=True)

# Metrics
MEASURES = [nDCG@10, MRR@10, Recall@100]

def load_local_qrels(path):
    """
    Reads the local qrels.json and converts it to the format ir_measures expects.
    Forces all IDs to strings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Qrels file not found at {path}. Did you run 01_prepare_data.py?")
    
    print(f"Loading qrels from {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    
    # Convert list of dicts to the iterator ir_measures wants
    # Structure: Qrel(query_id, doc_id, relevance)
    # CRITICAL: Force query_id and doc_id to strings!
    for item in data:
        yield ir_measures.Qrel(
            query_id=str(item['query_id']),
            doc_id=str(item['doc_id']),
            relevance=int(item['relevance'])
        )

def main():
    # 1. Load Ground Truth
    qrels = list(load_local_qrels(QRELS_PATH))
    
    # Debug: Print first qrel to check format
    print(f"Loaded {len(qrels)} relevance judgments.")
    print(f"Sample Qrel: {qrels[0]}")

    # 2. Find Run Files
    run_files = sorted(glob.glob(os.path.join(RUNS_DIR, "*.run")))
    if not run_files:
        print(f"❌ No .run files found in {RUNS_DIR}")
        return

    all_results = []
    print(f"Evaluating {len(run_files)} run files...")

    for rf in run_files:
        algo_name = os.path.basename(rf).replace(".run", "")
        print(f" -> Scoring {algo_name}...")
        
        try:
            # Load run file and FORCE IDs to strings
            run = ir_measures.read_trec_run(rf)
            
            # Calculate metrics
            metrics = ir_measures.calc_aggregate(MEASURES, qrels, run)
            
            # Format result
            row = {"Algorithm": algo_name}
            for measure, value in metrics.items():
                row[str(measure)] = value
            all_results.append(row)
            
        except Exception as e:
            print(f"   ⚠️ Error evaluating {algo_name}: {e}")

    # 3. Save & Print Leaderboard
    if not all_results:
        print("No results generated.")
        return

    df = pd.DataFrame(all_results)
    
    # Sort and Clean
    if "nDCG@10" in df.columns:
        df = df.sort_values(by="nDCG@10", ascending=False)
        cols = ["Algorithm", "nDCG@10", "MRR@10", "Recall@100"]
        df = df[[c for c in cols if c in df.columns]]

    print("\n" + "="*40)
    print("🏆 FINAL LEADERBOARD 🏆")
    print("="*40)
    print(df.to_string(index=False))
    
    # Save
    df.to_csv(os.path.join(RESULTS_DIR, "final_leaderboard.csv"), index=False)
    print(f"\nSaved results to {RESULTS_DIR}/final_leaderboard.csv")

if __name__ == "__main__":
    main()