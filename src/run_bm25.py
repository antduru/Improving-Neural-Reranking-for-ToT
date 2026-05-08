import argparse
import json
import re
from pathlib import Path
import wandb
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from load_data import load_tot_dataset
from metrics import evaluate_run

def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def run_bm25(
    dataset_name: str,
    k1: float,
    b: float,
    top_k: int,
    use_wandb: bool,
):
    queries, docs, qrels = load_tot_dataset(dataset_name)

    doc_ids = list(docs.keys())
    corpus = [tokenize(docs[doc_id]) for doc_id in tqdm(doc_ids, desc="Tokenizing documents")]

    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    rankings = dict()

    for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
        query_tokens = tokenize(query_text)
        scores = bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k:][::-1]
        rankings[query_id] = [doc_ids[i] for i in top_indices]

    metrics = evaluate_run(
        rankings=rankings,
        qrels=qrels,
        mrr_k=10,
        ndcg_k=10,
        recall_k=100,
    )

    result = {
        "dataset": dataset_name,
        "method": "BM25",
        "k1": k1,
        "b": b,
        "top_k": top_k,
        **metrics,
    }

    Path("../outputs").mkdir(exist_ok=True)

    output_path = Path("../outputs") / f"bm25_k1_{k1}_b_{b}_topk_{top_k}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    if use_wandb:
        wandb.init(
            project="is584-tot-reranking",
            name=f"bm25_k1_{k1}_b_{b}_topk_{top_k}",
            config={
                "dataset": dataset_name,
                "method": "BM25",
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