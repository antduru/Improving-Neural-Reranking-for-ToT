import math
from typing import Dict, List, Set


def reciprocal_rank(
    ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(
    ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 100
) -> float:
    if not relevant_doc_ids:
        return 0.0

    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved.intersection(relevant_doc_ids)) / len(relevant_doc_ids)


def ndcg_at_k(
    ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10
) -> float:
    dcg = 0.0

    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_relevant_count = min(len(relevant_doc_ids), k)
    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant_count + 1))
    return dcg / idcg


def evaluate_run(
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    mrr_k: int = 10,
    ndcg_k: int = 10,
    recall_k: int = 100,
) -> Dict[str, float]:
    rr_scores = []
    ndcg_scores = []
    recall_scores = []

    for query_id, relevant_doc_ids in qrels.items():
        ranked_doc_ids = rankings.get(query_id, [])

        rr_scores.append(reciprocal_rank(ranked_doc_ids, relevant_doc_ids, k=mrr_k))
        ndcg_scores.append(ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k=ndcg_k))
        recall_scores.append(recall_at_k(ranked_doc_ids, relevant_doc_ids, k=recall_k))

    return {
        f"mrr@{mrr_k}": sum(rr_scores) / len(rr_scores),
        f"ndcg@{ndcg_k}": sum(ndcg_scores) / len(ndcg_scores),
        f"recall@{recall_k}": sum(recall_scores) / len(recall_scores),
    }
