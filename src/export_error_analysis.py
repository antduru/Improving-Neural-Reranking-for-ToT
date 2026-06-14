import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.load_data import load_tot_dataset  # noqa: E402


def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ranking_list(value):
    if isinstance(value, dict):
        try:
            return [
                str(doc_id)
                for doc_id, _ in sorted(
                    value.items(),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
            ]
        except Exception:
            return [str(k) for k in value.keys()]

    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                doc_id = (
                    item.get("doc_id")
                    or item.get("docid")
                    or item.get("doc")
                    or item.get("id")
                )
                if doc_id is not None:
                    out.append(str(doc_id))
            else:
                out.append(str(item))
        return out

    return []


def normalize_rankings(obj):
    if isinstance(obj, dict):
        for key in ["rankings", "run", "results"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    if not isinstance(obj, dict):
        raise ValueError("Ranking JSON must be a dictionary or contain a rankings/run dictionary.")

    return {
        str(qid): normalize_ranking_list(ranking)
        for qid, ranking in obj.items()
    }


def get_rank(rankings, qid, doc_id):
    docs = rankings.get(str(qid), [])
    target = str(doc_id)

    for i, candidate_doc_id in enumerate(docs, start=1):
        if str(candidate_doc_id) == target:
            return i

    return None


def get_top_doc(rankings, qid):
    docs = rankings.get(str(qid), [])
    if not docs:
        return None
    return str(docs[0])


def get_relevant_doc_id(qrels, qid):
    rels = qrels.get(str(qid), {})
    if not rels:
        return None

    return max(rels.items(), key=lambda x: int(x[1]))[0]


def get_query_text(queries, qid):
    q = queries.get(str(qid), "")
    if isinstance(q, str):
        return q
    return getattr(q, "text", str(q))


def get_doc_title(docs, doc_id):
    if doc_id is None:
        return ""

    doc = docs.get(str(doc_id))
    if doc is None:
        return ""

    title = getattr(doc, "page_title", None)
    if title:
        return str(title)

    raw = getattr(doc, "raw", None)
    if isinstance(raw, dict):
        for key in ["page_title", "title", "name"]:
            if raw.get(key):
                return str(raw[key])

    text = str(doc)
    return text[:80].replace("\n", " ")


def rank_to_str(rank):
    return "" if rank is None else str(rank)


def assign_failure_mode(bm25_rank, ce_rank, fusion_rank):
    if bm25_rank is None:
        return "candidate_absent_from_bm25_1000"

    if bm25_rank > 100:
        if ce_rank is not None and ce_rank <= 100:
            return "ce_recovers_deep_candidate_into_top100"
        return "candidate_too_deep_for_top100"

    if bm25_rank <= 10:
        if ce_rank is not None and ce_rank > 10:
            if fusion_rank is not None and fusion_rank <= 10:
                return "ce_damages_top10_fusion_stabilizes"
            return "ce_damages_top10"

    if bm25_rank <= 100:
        if ce_rank is None or ce_rank > 100:
            return "ce_loses_top100_candidate"
        if ce_rank <= 10 and bm25_rank > 10:
            return "ce_promotes_candidate_to_top10"
        if fusion_rank is not None and fusion_rank <= 10 and bm25_rank > 10:
            return "fusion_promotes_candidate_to_top10"

    if fusion_rank is not None and ce_rank is not None:
        if ce_rank > bm25_rank and fusion_rank <= bm25_rank:
            return "fusion_controls_ce_demotion"

    return "stable_or_minor_rank_change"


def truncate(text, n=220):
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_representative_cases(rows):
    priority_modes = [
        "candidate_absent_from_bm25_1000",
        "candidate_too_deep_for_top100",
        "ce_recovers_deep_candidate_into_top100",
        "ce_damages_top10_fusion_stabilizes",
        "ce_damages_top10",
        "ce_loses_top100_candidate",
        "fusion_controls_ce_demotion",
        "ce_promotes_candidate_to_top10",
        "fusion_promotes_candidate_to_top10",
    ]

    selected = []
    used_qids = set()

    for mode in priority_modes:
        candidates = [r for r in rows if r["failure_mode"] == mode and r["query_id"] not in used_qids]
        if not candidates:
            continue

        def movement_score(r):
            ranks = []
            for key in ["bm25_rank", "pure_ce_rank", "fusion_rank"]:
                try:
                    ranks.append(int(r[key]))
                except Exception:
                    pass
            if len(ranks) < 2:
                return 0
            return max(ranks) - min(ranks)

        best = max(candidates, key=movement_score)
        selected.append(best)
        used_qids.add(best["query_id"])

    return selected[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bm25_rankings",
        default="outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json",
    )
    parser.add_argument(
        "--ce_rankings",
        required=True,
        help="Pure CrossEncoder ranking JSON.",
    )
    parser.add_argument(
        "--fusion_rankings",
        required=True,
        help="Rank-fusion CrossEncoder ranking JSON.",
    )
    parser.add_argument("--dataset_name", default="trec-tot/2023/dev")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--out_csv",
        default="outputs/error_analysis_rank_movements.csv",
    )
    parser.add_argument(
        "--out_examples_csv",
        default="outputs/error_analysis_representative_cases.csv",
    )
    args = parser.parse_args()

    print("[INFO] Loading dataset...")
    queries, docs, qrels = load_tot_dataset(args.dataset_name, args.data_dir)

    print("[INFO] Loading rankings...")
    bm25_rankings = normalize_rankings(load_json(args.bm25_rankings))
    ce_rankings = normalize_rankings(load_json(args.ce_rankings))
    fusion_rankings = normalize_rankings(load_json(args.fusion_rankings))

    rows = []

    for qid in sorted(qrels.keys(), key=lambda x: str(x)):
        rel_doc_id = get_relevant_doc_id(qrels, qid)
        if rel_doc_id is None:
            continue

        bm25_rank = get_rank(bm25_rankings, qid, rel_doc_id)
        ce_rank = get_rank(ce_rankings, qid, rel_doc_id)
        fusion_rank = get_rank(fusion_rankings, qid, rel_doc_id)

        bm25_top_doc = get_top_doc(bm25_rankings, qid)
        ce_top_doc = get_top_doc(ce_rankings, qid)
        fusion_top_doc = get_top_doc(fusion_rankings, qid)

        row = {
            "query_id": str(qid),
            "query_text": truncate(get_query_text(queries, qid), 260),
            "relevant_doc_id": str(rel_doc_id),
            "relevant_title": get_doc_title(docs, rel_doc_id),
            "bm25_rank": rank_to_str(bm25_rank),
            "pure_ce_rank": rank_to_str(ce_rank),
            "fusion_rank": rank_to_str(fusion_rank),
            "failure_mode": assign_failure_mode(bm25_rank, ce_rank, fusion_rank),
            "bm25_top1_doc_id": bm25_top_doc or "",
            "bm25_top1_title": get_doc_title(docs, bm25_top_doc),
            "pure_ce_top1_doc_id": ce_top_doc or "",
            "pure_ce_top1_title": get_doc_title(docs, ce_top_doc),
            "fusion_top1_doc_id": fusion_top_doc or "",
            "fusion_top1_title": get_doc_title(docs, fusion_top_doc),
        }
        rows.append(row)

    fieldnames = [
        "query_id",
        "query_text",
        "relevant_doc_id",
        "relevant_title",
        "bm25_rank",
        "pure_ce_rank",
        "fusion_rank",
        "failure_mode",
        "bm25_top1_doc_id",
        "bm25_top1_title",
        "pure_ce_top1_doc_id",
        "pure_ce_top1_title",
        "fusion_top1_doc_id",
        "fusion_top1_title",
    ]

    write_csv(args.out_csv, rows, fieldnames)

    examples = select_representative_cases(rows)
    write_csv(args.out_examples_csv, examples, fieldnames)

    print(f"[INFO] Wrote full rank-movement table: {args.out_csv}")
    print(f"[INFO] Wrote representative cases: {args.out_examples_csv}")

    counts = {}
    for row in rows:
        counts[row["failure_mode"]] = counts.get(row["failure_mode"], 0) + 1

    print("[INFO] Failure mode counts:")
    for mode, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()