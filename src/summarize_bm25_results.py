import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

def main():
    rows = []

    for path in OUTPUT_DIR.glob("rank_bm25_k1_*_b_*_topk_*.json"):
        with open(path, "r") as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["mrr@10", "ndcg@10", "recall@100"],
        ascending=False,
    )

    print(df.to_string(index=False))

    summary_path = OUTPUT_DIR / "bm25_sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()