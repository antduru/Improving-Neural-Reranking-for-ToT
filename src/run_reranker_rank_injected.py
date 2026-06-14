import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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


def load_rankings(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rankings = json.load(f)
    return {str(qid): [str(did) for did in dids] for qid, dids in rankings.items()}


def lexical_strength_from_rank(rank: int) -> str:
    if rank <= 10: return "very high"
    if rank <= 50: return "high"
    if rank <= 100: return "medium"
    if rank <= 500: return "low"
    return "very low"


def inject_rank_signal(doc_text: str, rank: int) -> str:
    return f"BM25 rank: {rank}. Lexical match strength: {lexical_strength_from_rank(rank)}. Document: " + doc_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrossEncoder reranking with rank signal injected into document text.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    queries, docs, qrels = load_tot_dataset(args.dataset)
    rankings = load_rankings(args.rankings_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model_name, device=device, max_length=512)
    reranked = {}

    for qid, ranked_doc_ids in tqdm(rankings.items(), desc="Rank-injected reranking"):
        candidates = ranked_doc_ids[:args.rerank_top_k]
        remaining = ranked_doc_ids[args.rerank_top_k:]
        pairs = [(queries[qid], inject_rank_signal(docs[doc_id], rank)) for rank, doc_id in enumerate(candidates, start=1)]
        scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
        order = np.argsort(scores)[::-1]
        reranked[qid] = [candidates[i] for i in order] + remaining
        
    metrics = evaluate_run(reranked, qrels)
    result = {"dataset": args.dataset, "method": "Rank-Injected CrossEncoder", "rankings_path": args.rankings_path, "model_name": args.model_name, "rerank_top_k": args.rerank_top_k, "batch_size": args.batch_size, "device": device, **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = f"rank_injected_{args.model_name.replace('/', '_')}_{Path(args.rankings_path).stem}_reranktop_{args.rerank_top_k}"
    (out / f"{prefix}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{prefix}_rankings.json").write_text(json.dumps(reranked, indent=2), encoding="utf-8")
    print("\nRank-injected reranking result:"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
