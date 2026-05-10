import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
import wandb
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from load_data import load_tot_dataset
from metrics import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def load_or_create_tokenized_corpus(docs):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "rank_bm25_tokenized_docs.pkl"

    if cache_path.exists():
        print(f"Loading cached tokenized corpus from {cache_path}")
        with open(cache_path, "rb") as f:
            doc_ids, corpus = pickle.load(f)
        return doc_ids, corpus

    print("Creating tokenized corpus cache...")
    doc_ids = list(docs.keys())

    corpus = [
        tokenize(docs[doc_id]) for doc_id in tqdm(doc_ids, desc="Tokenizing documents")
    ]

    with open(cache_path, "wb") as f:
        pickle.dump((doc_ids, corpus), f)

    return doc_ids, corpus


def get_top_k_doc_ids(scores, doc_ids, top_k: int):
    scores = np.asarray(scores)
    top_k = min(top_k, len(scores))

    top_indices_unsorted = np.argpartition(scores, -top_k)[-top_k:]
    top_indices_sorted = top_indices_unsorted[
        np.argsort(scores[top_indices_unsorted])[::-1]
    ]

    return [doc_ids[i] for i in top_indices_sorted]


def run_bm25(
    dataset_name: str,
    k1: float,
    b: float,
    top_k: int,
    use_wandb: bool,
):
    queries, docs, qrels = load_tot_dataset(dataset_name)
    doc_ids, corpus = load_or_create_tokenized_corpus(docs)
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    rankings = {}

    for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
        query_tokens = tokenize(query_text)
        scores = bm25.get_scores(query_tokens)
        rankings[query_id] = get_top_k_doc_ids(scores, doc_ids, top_k)

    metrics = evaluate_run(
        rankings=rankings,
        qrels=qrels,
        mrr_k=10,
        ndcg_k=10,
        recall_k=100,
    )

    result = {
        "dataset": dataset_name,
        "method": "rank-bm25",
        "k1": k1,
        "b": b,
        "top_k": top_k,
        **metrics,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)

    result_path = OUTPUT_DIR / f"rank_bm25_k1_{k1}_b_{b}_topk_{top_k}.json"
    rankings_path = OUTPUT_DIR / f"rank_bm25_rankings_k1_{k1}_b_{b}_topk_{top_k}.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    with open(rankings_path, "w") as f:
        json.dump(rankings, f)

    if use_wandb:
        wandb.init(
            project="is584-tot-reranking",
            name=f"rank_bm25_k1_{k1}_b_{b}_topk_{top_k}",
            config={
                "dataset": dataset_name,
                "method": "rank-bm25",
                "k1": k1,
                "b": b,
                "top_k": top_k,
            },
        )
        wandb.log(metrics)
        wandb.finish()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    run_bm25(
        dataset_name=args.dataset_name,
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
        use_wandb=args.use_wandb,
    )
