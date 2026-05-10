import argparse
import json
from pathlib import Path
import numpy as np
import torch
import wandb
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from load_data import load_tot_dataset
from metrics import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def rerank_with_cross_encoder(
    queries,
    docs,
    bm25_rankings,
    model_name,
    rerank_top_k,
    batch_size,
    device,
):
    model = CrossEncoder(model_name, device=device)

    reranked = {}

    for query_id, ranked_doc_ids in tqdm(bm25_rankings.items(), desc="Reranking queries"):
        candidate_doc_ids = ranked_doc_ids[:rerank_top_k]

        pairs = [
            (queries[query_id], docs[doc_id])
            for doc_id in candidate_doc_ids
        ]

        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(scores)

        sorted_indices = np.argsort(scores)[::-1]
        reranked_top = [candidate_doc_ids[i] for i in sorted_indices]

        remaining = ranked_doc_ids[rerank_top_k:]
        reranked[query_id] = reranked_top + remaining

    return reranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--bm25_rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries, docs, qrels = load_tot_dataset(args.dataset_name)
    bm25_rankings = load_json(args.bm25_rankings_path)

    reranked = rerank_with_cross_encoder(
        queries=queries,
        docs=docs,
        bm25_rankings=bm25_rankings,
        model_name=args.model_name,
        rerank_top_k=args.rerank_top_k,
        batch_size=args.batch_size,
        device=device,
    )

    metrics = evaluate_run(
        rankings=reranked,
        qrels=qrels,
        mrr_k=10,
        ndcg_k=10,
        recall_k=100,
    )

    result = {
        "dataset": args.dataset_name,
        "method": "BM25 + CrossEncoder",
        "bm25_rankings_path": args.bm25_rankings_path,
        "model_name": args.model_name,
        "rerank_top_k": args.rerank_top_k,
        "batch_size": args.batch_size,
        "device": device,
        **metrics,
    }

    safe_model_name = args.model_name.replace("/", "_")
    result_path = OUTPUT_DIR / f"reranker_{safe_model_name}_topk_{args.rerank_top_k}.json"
    rankings_path = OUTPUT_DIR / f"reranker_rankings_{safe_model_name}_topk_{args.rerank_top_k}.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(rankings_path, "w") as f:
        json.dump(reranked, f)

    if args.use_wandb:
        wandb.init(
            project="is584-tot-reranking",
            name=f"reranker_{safe_model_name}_topk_{args.rerank_top_k}",
            config={
                "dataset": args.dataset_name,
                "method": "BM25 + CrossEncoder",
                "model_name": args.model_name,
                "rerank_top_k": args.rerank_top_k,
                "batch_size": args.batch_size,
                "device": device,
            },
        )
        wandb.log(metrics)
        wandb.finish()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()