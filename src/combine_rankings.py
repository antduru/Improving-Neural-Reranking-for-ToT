import argparse
import json
from pathlib import Path
from typing import Dict, List

from load_data import load_tot_dataset
from metrics import evaluate_run


def load_rankings(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rankings = json.load(f)
    return {str(qid): [str(did) for did in dids] for qid, dids in rankings.items()}


def union_rankings(primary_rankings: Dict[str, List[str]], secondary_rankings: Dict[str, List[str]], primary_k: int, secondary_k: int, max_output_k: int) -> Dict[str, List[str]]:
    combined = {}
    for qid in sorted(set(primary_rankings) | set(secondary_rankings)):
        seen, merged = set(), []
        for doc_id in primary_rankings.get(qid, [])[:primary_k] + secondary_rankings.get(qid, [])[:secondary_k]:
            if doc_id not in seen:
                merged.append(doc_id); seen.add(doc_id)
        combined[qid] = merged[:max_output_k]
    return combined


def reciprocal_rank_fusion(rankings_a: Dict[str, List[str]], rankings_b: Dict[str, List[str]], a_k: int, b_k: int, rrf_k: int, max_output_k: int) -> Dict[str, List[str]]:
    combined = {}
    for qid in sorted(set(rankings_a) | set(rankings_b)):
        scores = {}
        for rank, doc_id in enumerate(rankings_a.get(qid, [])[:a_k], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, doc_id in enumerate(rankings_b.get(qid, [])[:b_k], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        combined[qid] = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:max_output_k]
        
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine two retrieval rankings using union or RRF.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--ranking_a", type=str, required=True)
    parser.add_argument("--ranking_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default="BM25")
    parser.add_argument("--label_b", type=str, default="Dense")
    parser.add_argument("--a_k", type=int, required=True)
    parser.add_argument("--b_k", type=int, required=True)
    parser.add_argument("--method", type=str, choices=["union", "rrf"], required=True)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--max_output_k", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--output_name", type=str, default=None)
    args = parser.parse_args()

    rankings_a = load_rankings(args.ranking_a)
    rankings_b = load_rankings(args.ranking_b)
    if args.method == "union":
        combined = union_rankings(rankings_a, rankings_b, args.a_k, args.b_k, args.max_output_k)
    else:
        combined = reciprocal_rank_fusion(rankings_a, rankings_b, args.a_k, args.b_k, args.rrf_k, args.max_output_k)
    _, _, qrels = load_tot_dataset(args.dataset)
    metrics = evaluate_run(combined, qrels)
    result = {"method": args.method, "dataset": args.dataset, "ranking_a": args.label_a, "ranking_b": args.label_b, "a_k": args.a_k, "b_k": args.b_k, "rrf_k": args.rrf_k if args.method == "rrf" else None, "max_output_k": args.max_output_k, **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.method}_{args.label_a}_k{args.a_k}_{args.label_b}_k{args.b_k}_topk_{args.max_output_k}"
    output_name = output_name.replace("/", "_").replace(" ", "_")
    rankings_path = out / f"{output_name}_rankings.json"
    result_path = out / f"{output_name}_result.json"
    rankings_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nCombined retrieval result:")
    print(json.dumps(result, indent=2))
    print(f"\nSaved rankings to: {rankings_path}")
    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()
