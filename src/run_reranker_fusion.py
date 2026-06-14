import argparse
import json
from pathlib import Path

import numpy as np
import torch
import wandb
import os
import random
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from load_data import load_tot_dataset
from metrics import evaluate_run
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--bm25_rankings_path", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_bm25", type=float, default=0.75)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries, docs, qrels = load_tot_dataset(args.dataset_name)
    bm25_rankings = load_json(args.bm25_rankings_path)

    model = CrossEncoder(args.model_name, device=device)

    fused_rankings = {}

    for query_id, ranked_doc_ids in tqdm(
        bm25_rankings.items(), desc="Fusion reranking"
    ):
        candidate_doc_ids = ranked_doc_ids[: args.rerank_top_k]

        pairs = [(queries[query_id], docs[doc_id]) for doc_id in candidate_doc_ids]

        ce_scores = model.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        ce_norm = minmax_normalize(ce_scores)

        # Rank-based BM25 signal: highest BM25-ranked document gets highest score.
        bm25_rank_scores = np.linspace(
            1.0,
            0.0,
            num=len(candidate_doc_ids),
            dtype=np.float32,
        )

        final_scores = (
            args.lambda_bm25 * bm25_rank_scores + (1.0 - args.lambda_bm25) * ce_norm
        )

        sorted_indices = np.argsort(final_scores)[::-1]
        fused_top = [candidate_doc_ids[i] for i in sorted_indices]

        remaining = ranked_doc_ids[args.rerank_top_k :]
        fused_rankings[query_id] = fused_top + remaining

    metrics = evaluate_run(
        rankings=fused_rankings,
        qrels=qrels,
        mrr_k=10,
        ndcg_k=10,
        recall_k=100,
    )

    result = {
        "dataset": args.dataset_name,
        "method": "BM25 Rank Fusion + CrossEncoder",
        "model_name": args.model_name,
        "rerank_top_k": args.rerank_top_k,
        "lambda_bm25": args.lambda_bm25,
        "device": device,
        **metrics,
    }

    safe_model_name = args.model_name.replace("/", "_")
    result_path = (
        OUTPUT_DIR
        / f"rank_fusion_{safe_model_name}_topk_{args.rerank_top_k}_lambda_{args.lambda_bm25}.json"
    )
    rankings_path = (
        OUTPUT_DIR
        / f"rank_fusion_rankings_{safe_model_name}_topk_{args.rerank_top_k}_lambda_{args.lambda_bm25}.json"
    )

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(rankings_path, "w") as f:
        json.dump(fused_rankings, f)

    if args.use_wandb:
        wandb.init(
            project="is584-tot-reranking",
            name=f"rank_fusion_lambda_{args.lambda_bm25}",
            config=result,
        )
        wandb.log(metrics)
        wandb.finish()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
