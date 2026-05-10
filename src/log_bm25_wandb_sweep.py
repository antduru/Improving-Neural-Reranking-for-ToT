import json
import re
from pathlib import Path
import wandb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

def parse_run_name(run_name):
    pattern = r"k1_(?P<k1>[0-9.]+)_b_(?P<b>[0-9.]+)_topk_(?P<top_k>[0-9]+)"
    match = re.match(pattern, run_name)
    if match is None:
        raise ValueError(f"Invalid run_name: {run_name}")

    return {
        "k1": float(match.group("k1")),
        "b": float(match.group("b")),
        "top_k": int(match.group("top_k")),
    }

def main():
    wandb.init(
        entity="ant-duru",
        project="is584-tot-reranking",
    )

    run_name = wandb.config.run_name
    parsed = parse_run_name(run_name)
    k1 = parsed["k1"]
    b = parsed["b"]
    top_k = parsed["top_k"]
    result_path = OUTPUT_DIR / f"rank_bm25_k1_{k1}_b_{b}_topk_{top_k}.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")

    with open(result_path, "r") as f:
        result = json.load(f)

    wandb.config.update(
        {
            "method": "rank-bm25",
            "k1": k1,
            "b": b,
            "top_k": top_k,
        },
        allow_val_change=True,
    )

    wandb.log(
        {
            "mrr@10": result["mrr@10"],
            "ndcg@10": result["ndcg@10"],
            "recall@100": result["recall@100"],
        }
    )

    wandb.finish()

if __name__ == "__main__":
    main()
