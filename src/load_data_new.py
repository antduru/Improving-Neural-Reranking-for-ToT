import json
from pathlib import Path
from collections import namedtuple


Query = namedtuple("Query", ["query_id", "text"])
Doc = namedtuple("Doc", ["doc_id", "text"])
Qrel = namedtuple("Qrel", ["query_id", "doc_id", "relevance"])


def _doc_to_text(obj):
    parts = []

    for key in ["page_title", "title", "name"]:
        value = obj.get(key)
        if value:
            parts.append(str(value))

    text = obj.get("text")
    if text:
        parts.append(str(text))

    sections = obj.get("sections")
    if isinstance(sections, dict):
        for section_name, section_text in sections.items():
            if section_name:
                parts.append(str(section_name))
            if section_text:
                parts.append(str(section_text))
    elif isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                heading = section.get("heading") or section.get("title") or section.get("section_title")
                section_text = section.get("text") or section.get("body")
                if heading:
                    parts.append(str(heading))
                if section_text:
                    parts.append(str(section_text))
            elif section:
                parts.append(str(section))

    infoboxes = obj.get("infoboxes")
    if isinstance(infoboxes, dict):
        for box_name, box_values in infoboxes.items():
            if box_name:
                parts.append(str(box_name))
            if isinstance(box_values, dict):
                for k, v in box_values.items():
                    if v:
                        parts.append(f"{k}: {v}")
            elif box_values:
                parts.append(str(box_values))
    elif isinstance(infoboxes, list):
        for box in infoboxes:
            if isinstance(box, dict):
                for k, v in box.items():
                    if v:
                        parts.append(f"{k}: {v}")
            elif box:
                parts.append(str(box))

    return " ".join(parts)


def _find_first_existing(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def _read_query_text(obj):
    for key in ["text", "query", "description", "title"]:
        if obj.get(key):
            return str(obj[key])
    return ""


def _read_query_id(obj):
    for key in ["query_id", "qid", "id", "queryId"]:
        if obj.get(key):
            return str(obj[key])
    raise ValueError(f"Could not find query id in object: {obj}")


def _read_doc_id(obj):
    for key in ["doc_id", "docid", "id", "page_id", "wikidata_id"]:
        if obj.get(key):
            return str(obj[key])
    raise ValueError(f"Could not find doc id in object keys: {list(obj.keys())}")


def _load_local_tot(data_dir):
    data_dir = Path(data_dir)

    corpus_path = _find_first_existing([
        data_dir / "corpus.jsonl",
        data_dir / "corpus" / "corpus.jsonl",
        data_dir / "TREC-ToT" / "corpus.jsonl",
    ])

    queries_path = _find_first_existing([
        data_dir / "dev" / "queries.jsonl",
        data_dir / "2023-dev" / "queries.jsonl",
        data_dir / "dev" / "2023-dev.jsonl",
        data_dir / "2023-dev.jsonl",
        data_dir / "queries.jsonl",
    ])

    qrels_path = _find_first_existing([
        data_dir / "2023-qrels.txt",
        data_dir / "qrels.txt",
        data_dir / "dev" / "qrels.txt",
        data_dir / "2023-dev" / "qrels.txt",
    ])

    if corpus_path is None:
        raise FileNotFoundError("Could not find corpus.jsonl under data/")
    if queries_path is None:
        raise FileNotFoundError("Could not find dev queries file under data/")
    if qrels_path is None:
        raise FileNotFoundError("Could not find qrels file under data/")

    print(f"[INFO] Loading local corpus from: {corpus_path}")
    print(f"[INFO] Loading local queries from: {queries_path}")
    print(f"[INFO] Loading local qrels from: {qrels_path}")

    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            query_id = _read_query_id(obj)
            text = _read_query_text(obj)
            queries[query_id] = Query(query_id=query_id, text=text)

    docs = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = _read_doc_id(obj)
            docs[doc_id] = _doc_to_text(obj)

            if (i + 1) % 50000 == 0:
                print(f"[INFO] Loaded {i + 1} documents...")

    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) >= 4:
                query_id = str(parts[0])
                doc_id = str(parts[2])
                relevance = int(parts[3])
            elif len(parts) == 3:
                query_id = str(parts[0])
                doc_id = str(parts[1])
                relevance = int(parts[2])
            else:
                raise ValueError(f"Unexpected qrels line format: {line}")

            qrels.setdefault(query_id, {})[doc_id] = relevance

    print(f"[INFO] Loaded {len(queries)} queries, {len(docs)} docs, {len(qrels)} qrel query entries.")

    return queries, docs, qrels


def load_tot_dataset(dataset_name="trec-tot/2023/dev", data_dir="data"):
    data_dir = Path(data_dir)

    has_local_data = (
        data_dir.exists()
        and any(data_dir.rglob("corpus.jsonl"))
        and (
            any(data_dir.rglob("queries.jsonl"))
            or any(data_dir.rglob("2023-dev.jsonl"))
        )
        and (
            (data_dir / "2023-qrels.txt").exists()
            or any(data_dir.rglob("qrels.txt"))
        )
    )

    if has_local_data:
        return _load_local_tot(data_dir)

    print("[INFO] Local data not found. Falling back to ir_datasets.")

    import ir_datasets

    dataset = ir_datasets.load(dataset_name)

    queries = {}
    for query in dataset.queries_iter():
        queries[str(query.query_id)] = query

    docs = {}
    for doc in dataset.docs_iter():
        docs[str(doc.doc_id)] = doc

    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = int(qrel.relevance)

    return queries, docs, qrels