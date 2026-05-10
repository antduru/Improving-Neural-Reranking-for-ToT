import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def main():
    rows = []

    patterns = [
        "rank_bm25_k1_*_b_*_topk_*.json",
        "reranker_cross-encoder_*_topk_*.json",
        "fusion_cross-encoder_*_lambda_*.json",
        "rank_fusion_cross-encoder_*_lambda_*.json",
    ]

    for pattern in patterns:
        for path in OUTPUT_DIR.glob(pattern):
            with open(path, "r") as f:
                row = json.load(f)
            row["file"] = path.name
            rows.append(row)

    df = pd.DataFrame(rows)

    keep_cols = [
        "method",
        "k1",
        "b",
        "top_k",
        "rerank_top_k",
        "lambda_bm25",
        "mrr@10",
        "ndcg@10",
        "recall@100",
        "file",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    df = df.sort_values(
        by=["mrr@10", "ndcg@10", "recall@100"],
        ascending=False,
    )

    print(df.to_string(index=False))
    out_path = OUTPUT_DIR / "all_results_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
