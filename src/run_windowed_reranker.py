import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from load_data import load_tot_dataset
from metrics import evaluate_run


def load_rankings(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rankings = json.load(f)
    return {str(qid): [str(did) for did in dids] for qid, dids in rankings.items()}


def windowed_rerank_doc_ids(query_text: str, ranked_doc_ids: List[str], docs: Dict[str, str], model: CrossEncoder, rerank_top_k: int, window_size: int, batch_size: int) -> List[str]:
    top_doc_ids = ranked_doc_ids[:rerank_top_k]
    remaining = ranked_doc_ids[rerank_top_k:]
    reranked_top = []

    for start in range(0, len(top_doc_ids), window_size):
        window_doc_ids = top_doc_ids[start:start + window_size]
        pairs = [(query_text, docs[doc_id]) for doc_id in window_doc_ids]
        scores = np.asarray(model.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)
        order = np.argsort(scores)[::-1]
        reranked_top.extend([window_doc_ids[i] for i in order])
        
    return reranked_top + remaining


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local/windowed CrossEncoder reranking.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    queries, docs, qrels = load_tot_dataset(args.dataset)
    rankings = load_rankings(args.rankings_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model_name, device=device, max_length=512)
    reranked = {}

    for qid, ranked_doc_ids in tqdm(rankings.items(), desc="Windowed reranking"):
        reranked[qid] = windowed_rerank_doc_ids(queries[qid], ranked_doc_ids, docs, model, args.rerank_top_k, args.window_size, args.batch_size)

    metrics = evaluate_run(reranked, qrels)
    result = {"dataset": args.dataset, "method": "Windowed CrossEncoder Reranking", "rankings_path": args.rankings_path, "model_name": args.model_name, "rerank_top_k": args.rerank_top_k, "window_size": args.window_size, "batch_size": args.batch_size, "device": device, **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = f"windowed_{args.model_name.replace('/', '_')}_{Path(args.rankings_path).stem}_top{args.rerank_top_k}_window{args.window_size}"
    (out / f"{prefix}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{prefix}_rankings.json").write_text(json.dumps(reranked, indent=2), encoding="utf-8")
    print("\nWindowed reranking result:"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
