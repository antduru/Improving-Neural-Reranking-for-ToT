import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Set

import ir_datasets


def load_qrels(dataset_name: str) -> Dict[str, Set[str]]:
    dataset = ir_datasets.load(dataset_name)
    qrels: Dict[str, Set[str]] = {}
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels.setdefault(str(qrel.query_id), set()).add(str(qrel.doc_id))
    return qrels


def load_rankings(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rankings = json.load(f)
    return {str(qid): [str(did) for did in dids] for qid, dids in rankings.items()}


def rr_at_k(ranked: List[str], rels: Set[str], k: int) -> float:
    for rank, did in enumerate(ranked[:k], start=1):
        if did in rels:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: List[str], rels: Set[str], k: int) -> float:
    for rank, did in enumerate(ranked[:k], start=1):
        if did in rels:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def recall_at_k(ranked: List[str], rels: Set[str], k: int) -> float:
    return float(bool(set(ranked[:k]).intersection(rels)))


def oracle_ranking(ranked: List[str], rels: Set[str], candidate_k: int) -> List[str]:
    candidates = ranked[:candidate_k]
    relevant = [did for did in candidates if did in rels]
    if not relevant:
        return candidates
    chosen = relevant[0]
    return [chosen] + [did for did in candidates if did != chosen]


def evaluate(rankings: Dict[str, List[str]], qrels: Dict[str, Set[str]], candidate_k: int, eval_k: int) -> Dict[str, float]:
    actual_mrr = actual_ndcg = actual_recall = 0.0
    oracle_mrr = oracle_ndcg = oracle_recall = 0.0
    available = 0
    for qid, rels in qrels.items():
        ranked = rankings.get(qid, [])
        actual_mrr += rr_at_k(ranked, rels, eval_k)
        actual_ndcg += ndcg_at_k(ranked, rels, eval_k)
        actual_recall += recall_at_k(ranked, rels, eval_k)
        if set(ranked[:candidate_k]).intersection(rels):
            available += 1
        oracle = oracle_ranking(ranked, rels, candidate_k)
        oracle_mrr += rr_at_k(oracle, rels, eval_k)
        oracle_ndcg += ndcg_at_k(oracle, rels, eval_k)
        oracle_recall += recall_at_k(oracle, rels, eval_k)
        
    n = len(qrels)
    row = {"candidate_k": candidate_k, "eval_k": eval_k, "candidate_recall": available / n, "actual_mrr": actual_mrr / n, "actual_ndcg": actual_ndcg / n, "actual_recall": actual_recall / n, "oracle_mrr": oracle_mrr / n, "oracle_ndcg": oracle_ndcg / n, "oracle_recall": oracle_recall / n, "num_queries": n}
    row["mrr_gap"] = row["oracle_mrr"] - row["actual_mrr"]
    row["ndcg_gap"] = row["oracle_ndcg"] - row["actual_ndcg"]
    row["recall_gap"] = row["oracle_recall"] - row["actual_recall"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute actual vs oracle upper-bound metrics for candidate pools.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings", type=str, required=True)
    parser.add_argument("--candidate_ks", type=str, required=True)
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    if not Path(args.rankings).exists():
        raise FileNotFoundError(args.rankings)
    qrels = load_qrels(args.dataset)
    rankings = load_rankings(args.rankings)
    rows = [evaluate(rankings, qrels, int(k.strip()), args.eval_k) for k in args.candidate_ks.split(",") if k.strip()]
    print("\nOracle upper-bound analysis:")

    for r in rows:
        print(f"K={r['candidate_k']} | candidate_recall={r['candidate_recall']:.4f} | actual_mrr={r['actual_mrr']:.4f} | oracle_mrr={r['oracle_mrr']:.4f} | gap={r['mrr_gap']:.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
