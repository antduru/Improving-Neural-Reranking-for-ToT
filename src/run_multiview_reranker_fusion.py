import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import ir_datasets
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

IMPORTANT_INFOBOX_KEYS = {"name","native_name","director","producer","writer","screenplay","based_on","starring","music","cinematography","editing","production_companies","distributor","released","runtime","country","language","genre","author","publisher","developer","platform","artist","album","created_by","original_network","composer"}


def load_rankings(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        rankings = json.load(f)
    return {str(qid): [str(did) for did in dids] for qid, dids in rankings.items()}


def clean_wiki_markup(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    replacements = {"[[":"", "]]":"", "{{":"", "}}":"", "'''":"", "''":"", "<br>":" ", "<br/>":" ", "<br />":" ", "|":" "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def extract_infobox_summary(infoboxes: Any, max_items: int = 16) -> str:
    if not infoboxes:
        return ""
    parts = []
    for infobox in infoboxes:
        params = infobox.get("params", {}) if isinstance(infobox, dict) else {}
        for key, value in params.items():
            key_clean = str(key).strip()
            if key_clean not in IMPORTANT_INFOBOX_KEYS:
                continue
            value_clean = clean_wiki_markup(value)
            if not value_clean:
                continue
            parts.append(f"{key_clean.replace('_', ' ')}: {value_clean}")
            if len(parts) >= max_items:
                break
        if len(parts) >= max_items:
            break
    return "; ".join(parts)


def build_doc_views(doc: Any) -> Dict[str, str]:
    title = clean_wiki_markup(getattr(doc, "page_title", ""))
    full_text = clean_wiki_markup(getattr(doc, "text", ""))
    sections = getattr(doc, "sections", {}) or {}
    abstract = clean_wiki_markup(sections.get("abstract", ""))
    synopsis = clean_wiki_markup(sections.get("synopsis", ""))
    reception = clean_wiki_markup(sections.get("reception", ""))
    metadata = extract_infobox_summary(getattr(doc, "infoboxes", []), max_items=16)
    views = {}
    if full_text: views["full_text"] = full_text
    title_abstract = []
    if title: title_abstract.append(f"Title: {title}.")
    if abstract: title_abstract.append(f"Abstract: {abstract}")
    if title_abstract: views["title_abstract"] = "\n".join(title_abstract)
    if synopsis: views["synopsis"] = f"Synopsis: {synopsis}"
    if metadata: views["metadata"] = f"Metadata: {metadata}."
    compact = []
    if title: compact.append(f"Title: {title}.")
    if abstract: compact.append(f"Abstract: {abstract}")
    if synopsis: compact.append(f"Synopsis: {synopsis}")
    if metadata: compact.append(f"Metadata: {metadata}.")
    if compact: views["compact"] = "\n".join(compact)
    if reception: views["reception"] = f"Reception: {reception}"
    return views or {"empty": ""}


def load_multiview_docs(dataset_name: str) -> Dict[str, Dict[str, str]]:
    dataset = ir_datasets.load(dataset_name)
    docs = {}
    for doc in tqdm(dataset.docs_iter(), desc="Building multiview documents"):
        docs[str(doc.doc_id)] = build_doc_views(doc)
    return docs


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if float(np.max(scores)) - float(np.min(scores)) < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - float(np.min(scores))) / (float(np.max(scores)) - float(np.min(scores)))


def aggregate_view_scores(view_scores: List[float], aggregation: str) -> float:
    scores = np.asarray(view_scores, dtype=np.float32)
    if aggregation == "max": return float(np.max(scores))
    if aggregation == "mean": return float(np.mean(scores))
    if aggregation == "max_mean": return float(0.5 * np.max(scores) + 0.5 * np.mean(scores))
    raise ValueError(f"Unknown aggregation: {aggregation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiview CrossEncoder reranking with rank-preserving fusion.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_rank", type=float, default=0.99)
    parser.add_argument("--aggregation", type=str, choices=["max", "mean", "max_mean"], default="max")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    queries, _, qrels = load_tot_dataset(args.dataset)
    docs_by_view = load_multiview_docs(args.dataset)
    rankings = load_rankings(args.rankings_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model_name, device=device, max_length=512)
    reranked = {}
    for qid, ranked_doc_ids in tqdm(rankings.items(), desc="Multiview fusion reranking"):
        candidates = ranked_doc_ids[:args.rerank_top_k]
        remaining = ranked_doc_ids[args.rerank_top_k:]
        pairs, pair_doc_indices = [], []
        for doc_idx, doc_id in enumerate(candidates):
            for view_text in docs_by_view.get(doc_id, {"empty": ""}).values():
                pairs.append((queries[qid], view_text)); pair_doc_indices.append(doc_idx)
        raw_scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False), dtype=np.float32)
        scores_by_doc = [[] for _ in candidates]
        for score, doc_idx in zip(raw_scores, pair_doc_indices):
            scores_by_doc[doc_idx].append(float(score))
        ce_scores = np.asarray([aggregate_view_scores(scores, args.aggregation) for scores in scores_by_doc], dtype=np.float32)
        ce_norm = min_max_normalize(ce_scores)
        rank_scores = np.linspace(1.0, 0.0, num=len(candidates), dtype=np.float32)
        final_scores = args.lambda_rank * rank_scores + (1.0 - args.lambda_rank) * ce_norm
        order = np.argsort(final_scores)[::-1]
        reranked[qid] = [candidates[i] for i in order] + remaining
    metrics = evaluate_run(reranked, qrels)
    result = {"dataset": args.dataset, "method": "Multiview CrossEncoder + Rank Fusion", "rankings_path": args.rankings_path, "model_name": args.model_name, "rerank_top_k": args.rerank_top_k, "batch_size": args.batch_size, "device": device, "lambda_rank": args.lambda_rank, "aggregation": args.aggregation, "views": ["full_text", "title_abstract", "synopsis", "metadata", "compact", "reception"], **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = f"multiview_{args.model_name.replace('/', '_')}_{Path(args.rankings_path).stem}_top{args.rerank_top_k}_lambda{args.lambda_rank}_{args.aggregation}".replace('.', 'p')
    (out / f"{prefix}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{prefix}_rankings.json").write_text(json.dumps(reranked, indent=2), encoding="utf-8")
    print("\nMultiview fusion reranking result:"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
