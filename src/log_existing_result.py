import json
import argparse
import wandb


def main(result_path: str):
    with open(result_path, "r") as f:
        result = json.load(f)

    config = {
        "dataset": result["dataset"],
        "method": result["method"],
        "k1": result["k1"],
        "b": result["b"],
        "top_k": result["top_k"],
    }

    metrics = {
        "mrr@10": result["mrr@10"],
        "ndcg@10": result["ndcg@10"],
        "recall@100": result["recall@100"],
    }

    wandb.init(
        project="is584-tot-reranking",
        name=f"bm25_k1_{result['k1']}_b_{result['b']}_topk_{result['top_k']}",
        config=config,
    )

    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    main(args.result_path)
