import argparse
import json
from pathlib import Path
from load_data import load_tot_dataset
from metrics import recall_at_k


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def mean_recall_at_k(rankings, qrels, k):
    scores = []
    for query_id, relevant_doc_ids in qrels.items():
        ranked_doc_ids = rankings.get(query_id, [])
        scores.append(recall_at_k(ranked_doc_ids, relevant_doc_ids, k=k))

    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)

    args = parser.parse_args()

    _, _, qrels = load_tot_dataset(args.dataset_name)
    rankings = load_json(args.rankings_path)

    result = {
        "rankings_path": args.rankings_path,
        "recall@100": mean_recall_at_k(rankings, qrels, 100),
        "recall@500": mean_recall_at_k(rankings, qrels, 500),
        "recall@1000": mean_recall_at_k(rankings, qrels, 1000),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
