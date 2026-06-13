import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from load_data import load_tot_dataset
from metrics import evaluate_run


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, normalize: bool, show_progress_bar: bool = True) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize, show_progress_bar=show_progress_bar)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings if normalize else normalize_embeddings(embeddings)


def get_top_k_doc_ids(scores: np.ndarray, doc_ids: List[str], top_k: int) -> List[str]:
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices_unsorted = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices_unsorted[np.argsort(scores[top_indices_unsorted])[::-1]]
    return [doc_ids[i] for i in top_indices]


def run_dense_retrieval(dataset_name: str, model_name: str, top_k: int, batch_size: int, output_dir: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    queries, docs, qrels = load_tot_dataset(dataset_name)
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    doc_ids = list(docs.keys())
    doc_texts = [docs[did] for did in doc_ids]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading dense model: {model_name}")
    print(f"Device: {device}")
    model = SentenceTransformer(model_name, device=device)
    print(f"Encoding {len(doc_texts)} documents...")
    doc_embeddings = encode_texts(model, doc_texts, batch_size, normalize=True, show_progress_bar=True)
    print(f"Encoding {len(query_texts)} queries...")
    query_embeddings = encode_texts(model, query_texts, batch_size, normalize=True, show_progress_bar=True)
    rankings: Dict[str, List[str]] = {}
    print("Computing dense rankings...")
    for query_id, query_embedding in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
        scores = np.dot(doc_embeddings, query_embedding)
        rankings[query_id] = get_top_k_doc_ids(scores, doc_ids, top_k)
    metrics = evaluate_run(rankings, qrels)
    result = {"method": "dense", "dataset": dataset_name, "model_name": model_name, "top_k": top_k, "batch_size": batch_size, **metrics}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace("/", "_")
    result_file = out / f"dense_{safe_model_name}_topk_{top_k}.json"
    rankings_file = out / f"dense_rankings_{safe_model_name}_topk_{top_k}.json"
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rankings_file.write_text(json.dumps(rankings, indent=2), encoding="utf-8")
    print("\nDense retrieval result:")
    print(json.dumps(result, indent=2))
    print(f"\nSaved result to: {result_file}")
    print(f"Saved rankings to: {rankings_file}")
    return result, rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense retrieval for TREC-ToT.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/msmarco-MiniLM-L-6-v3")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    run_dense_retrieval(args.dataset, args.model_name, args.top_k, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
