import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from load_data import load_tot_dataset


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def reciprocal_rank(ranked_doc_ids, relevant_doc_ids, k=10):
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k=10):
    dcg = 0.0

    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_relevant_count = min(len(relevant_doc_ids), k)
    if ideal_relevant_count == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant_count + 1))
    return dcg / idcg


def win_tie_loss(base_scores, proposed_scores):
    wins = int(np.sum(proposed_scores > base_scores))
    ties = int(np.sum(proposed_scores == base_scores))
    losses = int(np.sum(proposed_scores < base_scores))
    return wins, ties, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--baseline_rankings", type=str, required=True)
    parser.add_argument("--proposed_rankings", type=str, required=True)
    parser.add_argument(
        "--out_path", type=str, default="../outputs/statistical_test.json"
    )
    args = parser.parse_args()

    _, _, qrels = load_tot_dataset(args.dataset_name)

    baseline = load_json(args.baseline_rankings)
    proposed = load_json(args.proposed_rankings)

    base_rr = []
    prop_rr = []

    base_ndcg = []
    prop_ndcg = []

    for query_id, relevant_doc_ids in qrels.items():
        base_ranking = baseline.get(query_id, [])
        prop_ranking = proposed.get(query_id, [])

        base_rr.append(reciprocal_rank(base_ranking, relevant_doc_ids, k=10))
        prop_rr.append(reciprocal_rank(prop_ranking, relevant_doc_ids, k=10))

        base_ndcg.append(ndcg_at_k(base_ranking, relevant_doc_ids, k=10))
        prop_ndcg.append(ndcg_at_k(prop_ranking, relevant_doc_ids, k=10))

    base_rr = np.array(base_rr)
    prop_rr = np.array(prop_rr)

    base_ndcg = np.array(base_ndcg)
    prop_ndcg = np.array(prop_ndcg)

    rr_stat, rr_p = wilcoxon(prop_rr, base_rr, zero_method="zsplit")
    ndcg_stat, ndcg_p = wilcoxon(prop_ndcg, base_ndcg, zero_method="zsplit")

    rr_w, rr_t, rr_l = win_tie_loss(base_rr, prop_rr)
    ndcg_w, ndcg_t, ndcg_l = win_tie_loss(base_ndcg, prop_ndcg)

    result = {
        "comparison": "baseline vs proposed",
        "rr@10": {
            "baseline_mean": float(base_rr.mean()),
            "proposed_mean": float(prop_rr.mean()),
            "delta": float(prop_rr.mean() - base_rr.mean()),
            "wilcoxon_statistic": float(rr_stat),
            "p_value": float(rr_p),
            "wins": rr_w,
            "ties": rr_t,
            "losses": rr_l,
        },
        "ndcg@10": {
            "baseline_mean": float(base_ndcg.mean()),
            "proposed_mean": float(prop_ndcg.mean()),
            "delta": float(prop_ndcg.mean() - base_ndcg.mean()),
            "wilcoxon_statistic": float(ndcg_stat),
            "p_value": float(ndcg_p),
            "wins": ndcg_w,
            "ties": ndcg_t,
            "losses": ndcg_l,
        },
    }

    print(json.dumps(result, indent=2))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
