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


def extract_infobox_summary(infoboxes: Any, max_items: int = 12) -> str:
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


def build_field_aware_doc_text(doc: Any) -> str:
    title = clean_wiki_markup(getattr(doc, "page_title", ""))
    sections = getattr(doc, "sections", {}) or {}
    abstract = clean_wiki_markup(sections.get("abstract", ""))
    synopsis = clean_wiki_markup(sections.get("synopsis", ""))
    metadata = extract_infobox_summary(getattr(doc, "infoboxes", []), max_items=12)
    full_text = clean_wiki_markup(getattr(doc, "text", ""))
    parts = []
    if title: parts.append(f"Title: {title}.")
    if abstract: parts.append(f"Abstract: {abstract}")
    if synopsis: parts.append(f"Synopsis: {synopsis}")
    if metadata: parts.append(f"Metadata: {metadata}.")
    if not parts and full_text: parts.append(f"Document: {full_text[:2000]}")
    return "\n".join(parts)


def load_field_aware_docs(dataset_name: str) -> Dict[str, str]:
    dataset = ir_datasets.load(dataset_name)
    docs = {}

    for doc in tqdm(dataset.docs_iter(), desc="Building field-aware documents"):
        docs[str(doc.doc_id)] = build_field_aware_doc_text(doc)
        
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run field-aware CrossEncoder reranking for TREC-ToT.")
    parser.add_argument("--dataset", type=str, default="trec-tot/2023/dev")
    parser.add_argument("--rankings_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank_top_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    queries, _, qrels = load_tot_dataset(args.dataset)
    field_docs = load_field_aware_docs(args.dataset)
    rankings = load_rankings(args.rankings_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.model_name, device=device, max_length=512)
    reranked = {}

    for qid, ranked_doc_ids in tqdm(rankings.items(), desc="Field-aware reranking"):
        candidates = ranked_doc_ids[:args.rerank_top_k]
        remaining = ranked_doc_ids[args.rerank_top_k:]
        pairs = [(queries[qid], field_docs.get(doc_id, "")) for doc_id in candidates]
        scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False), dtype=np.float32)
        order = np.argsort(scores)[::-1]
        reranked[qid] = [candidates[i] for i in order] + remaining

    metrics = evaluate_run(reranked, qrels)
    result = {"dataset": args.dataset, "method": "Field-Aware CrossEncoder", "rankings_path": args.rankings_path, "model_name": args.model_name, "rerank_top_k": args.rerank_top_k, "batch_size": args.batch_size, "device": device, "document_representation": "title + abstract + synopsis + selected infobox metadata", **metrics}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    prefix = f"field_aware_{args.model_name.replace('/', '_')}_{Path(args.rankings_path).stem}_reranktop_{args.rerank_top_k}"
    (out / f"{prefix}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{prefix}_rankings.json").write_text(json.dumps(reranked, indent=2), encoding="utf-8")
    print("\nField-aware reranking result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
