import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import ir_datasets
from scipy.stats import binomtest


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


def availability_vector(rankings: Dict[str, List[str]], qrels: Dict[str, Set[str]], candidate_k: int) -> Dict[str, int]:
    return {qid: int(bool(set(rankings.get(qid, [])[:candidate_k]).intersection(rels))) for qid, rels in qrels.items()}


def exact_mcnemar_p_value(baseline_availability: Dict[str, int], system_availability: Dict[str, int]) -> Tuple[float, int, int, int, int]:
    both = baseline_only = system_only = neither = 0
    for qid in sorted(set(baseline_availability) & set(system_availability)):
        b, s = baseline_availability[qid], system_availability[qid]
        if b == 1 and s == 1: both += 1
        elif b == 1 and s == 0: baseline_only += 1
        elif b == 0 and s == 1: system_only += 1
        else: neither += 1
    discordant = baseline_only + system_only
    p_value = 1.0 if discordant == 0 else binomtest(min(baseline_only, system_only), discordant, p=0.5, alternative="two-sided").pvalue
    
    return p_value, both, baseline_only, system_only, neither


def parse_system_arguments(system_args: List[str]) -> Dict[str, str]:
    systems = {}
    for item in system_args:
        if "=" not in item:
            raise ValueError(f"Invalid --system argument: {item}. Expected LABEL=path")
        label, path = item.split("=", 1)
        label, path = label.strip(), path.strip()
        if not Path(path).exists():
            raise FileNotFoundError(f"Ranking file not found: {path}")
        systems[label] = path
    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate recall with exact McNemar tests.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--baseline_label", type=str, required=True)
    parser.add_argument("--baseline_rankings", type=str, required=True)
    parser.add_argument("--baseline_k", type=int, required=True)
    parser.add_argument("--system", type=str, nargs="+", required=True, help="LABEL=path pairs")
    parser.add_argument("--candidate_ks", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    qrels = load_qrels(args.dataset)
    baseline_rankings = load_rankings(args.baseline_rankings)
    baseline_av = availability_vector(baseline_rankings, qrels, args.baseline_k)
    baseline_recall = sum(baseline_av.values()) / len(baseline_av)
    systems = parse_system_arguments(args.system)
    candidate_ks = [int(k.strip()) for k in args.candidate_ks.split(",") if k.strip()]
    rows = []
    for label, path in systems.items():
        rankings = load_rankings(path)
        for k in candidate_ks:
            sys_av = availability_vector(rankings, qrels, k)
            sys_recall = sum(sys_av.values()) / len(sys_av)
            p, both, baseline_only, system_only, neither = exact_mcnemar_p_value(baseline_av, sys_av)
            rows.append({"baseline": args.baseline_label, "baseline_k": args.baseline_k, "baseline_candidate_recall": round(baseline_recall, 6), "system": label, "candidate_k": k, "system_candidate_recall": round(sys_recall, 6), "delta_vs_baseline": round(sys_recall - baseline_recall, 6), "mcnemar_p_value": round(p, 6), "both_available": both, "baseline_only": baseline_only, "system_only": system_only, "neither_available": neither, "num_queries": len(baseline_av)})
    print("\nCandidate recall comparison:")
    
    for r in rows:
        print(f"{r['system']}@{r['candidate_k']} | Recall={r['system_candidate_recall']:.4f} | Δ={r['delta_vs_baseline']:.4f} | p={r['mcnemar_p_value']:.6f} | system_only={r['system_only']} | baseline_only={r['baseline_only']}")
    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
