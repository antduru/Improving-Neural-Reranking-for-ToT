import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if float(np.max(scores)) - float(np.min(scores)) < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - float(np.min(scores))) / (float(np.max(scores)) - float(np.min(scores)))


def gated_promote_ranking(candidate_doc_ids: List[str], ce_scores: np.ndarray, max_promotions: int, min_ce_norm: float, min_gap: float, max_original_rank: int) -> Tuple[List[str], Dict[str, float]]:
    ce_norm = min_max_normalize(ce_scores)
    ce_order = np.argsort(ce_norm)[::-1]
    promoted, promoted_set = [], set()
    
    for pos, idx in enumerate(ce_order):
        original_rank = idx + 1
        score = float(ce_norm[idx])
        if original_rank > max_original_rank or score < min_ce_norm:
            continue
        gap = score - float(ce_norm[ce_order[pos + 1]]) if pos + 1 < len(ce_order) else score
        if gap < min_gap:
            continue
        doc_id = candidate_doc_ids[idx]
        promoted.append(doc_id)
        promoted_set.add(doc_id)
        if len(promoted) >= max_promotions:
            break
    remaining = [doc_id for doc_id in candidate_doc_ids if doc_id not in promoted_set]
    return promoted + remaining, {"num_promoted": float(len(promoted)), "max_ce_norm": float(np.max(ce_norm)) if len(ce_norm) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative confidence-gated CrossEncoder reranking.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_promotions", type=int, default=3)
    parser.add_argument("--min_ce_norm", type=float, default=0.95)
    parser.add_argument("--min_gap", type=float, default=0.05)
    parser.add_argument("--max_original_rank", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    queries, docs, qrels = load_tot_dataset(args.dataset)
    rankings = load_rankings(args.rankings_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model_name, device=device, max_length=512)

    reranked = {}
    total_promoted = 0
    for qid, ranked_doc_ids in tqdm(rankings.items(), desc="Gated reranking"):
        candidates = ranked_doc_ids[:args.rerank_top_k]
        remaining = ranked_doc_ids[args.rerank_top_k:]
        pairs = [(queries[qid], docs[doc_id]) for doc_id in candidates]
        ce_scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False), dtype=np.float32)
        gated_top, info = gated_promote_ranking(candidates, ce_scores, args.max_promotions, args.min_ce_norm, args.min_gap, args.max_original_rank)
        total_promoted += int(info["num_promoted"])
        reranked[qid] = gated_top + remaining

    metrics = evaluate_run(reranked, qrels)
    result = {"dataset": args.dataset, "method": "Confidence-Gated CrossEncoder", "rankings_path": args.rankings_path, "model_name": args.model_name, "rerank_top_k": args.rerank_top_k, "batch_size": args.batch_size, "device": device, "max_promotions": args.max_promotions, "min_ce_norm": args.min_ce_norm, "min_gap": args.min_gap, "max_original_rank": args.max_original_rank, "total_promoted": total_promoted, "avg_promoted_per_query": total_promoted / max(len(rankings), 1), **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = f"gated_{args.model_name.replace('/', '_')}_{Path(args.rankings_path).stem}_top{args.rerank_top_k}_prom{args.max_promotions}_ce{args.min_ce_norm}_gap{args.min_gap}_maxrank{args.max_original_rank}".replace('.', 'p')
    (out / f"{prefix}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{prefix}_rankings.json").write_text(json.dumps(reranked, indent=2), encoding="utf-8")
    print("\nGated reranking result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
