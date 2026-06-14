import json
from pathlib import Path
from collections import namedtuple


Query = namedtuple("Query", ["query_id", "text"])


class DocText(str):
    """
    String-like document object.

    This behaves like a normal string for BM25/tokenization code:
        tokenize(docs[doc_id]) -> works

    But it also keeps structured fields for field-aware or multiview reranking:
        docs[doc_id].page_title
        docs[doc_id].sections
        docs[doc_id].infoboxes
    """

    def __new__(
        cls,
        text,
        doc_id=None,
        page_title=None,
        sections=None,
        infoboxes=None,
        raw=None,
    ):
        obj = str.__new__(cls, text or "")
        obj.doc_id = doc_id
        obj.text = text or ""
        obj.page_title = page_title or ""
        obj.sections = sections or {}
        obj.infoboxes = infoboxes or {}
        obj.raw = raw or {}
        return obj


def _safe_get(obj, keys, default=None):
    for key in keys:
        value = obj.get(key)
        if value is not None and value != "":
            return value
    return default


def _read_query_id(obj):
    query_id = _safe_get(obj, ["query_id", "qid", "id", "queryId"])
    if query_id is None:
        raise ValueError(f"Could not find query id in object: {obj}")
    return str(query_id)


def _read_query_text(obj):
    text = _safe_get(obj, ["text", "query", "description", "title"], "")
    return str(text)


def _read_doc_id(obj):
    doc_id = _safe_get(obj, ["doc_id", "docid", "id", "page_id", "wikidata_id"])
    if doc_id is None:
        raise ValueError(f"Could not find doc id in object keys: {list(obj.keys())}")
    return str(doc_id)


def _append_sections(parts, sections):
    if isinstance(sections, dict):
        for section_name, section_text in sections.items():
            if section_name:
                parts.append(str(section_name))
            if section_text:
                parts.append(str(section_text))

    elif isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                heading = _safe_get(section, ["heading", "title", "section_title", "name"])
                body = _safe_get(section, ["text", "body", "contents", "content"])

                if heading:
                    parts.append(str(heading))
                if body:
                    parts.append(str(body))
            elif section:
                parts.append(str(section))


def _append_infoboxes(parts, infoboxes):
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


def _doc_to_text(obj):
    parts = []

    page_title = _safe_get(obj, ["page_title", "title", "name"])
    if page_title:
        parts.append(str(page_title))

    main_text = _safe_get(obj, ["text", "body", "contents", "content"])
    if main_text:
        parts.append(str(main_text))

    return " ".join(parts)

def _make_doc(obj):
    doc_id = _read_doc_id(obj)
    page_title = _safe_get(obj, ["page_title", "title", "name"], "")
    text = _doc_to_text(obj)

    return doc_id, DocText(
        text=text,
        doc_id=doc_id,
        page_title=str(page_title or ""),
        sections=obj.get("sections") or {},
        infoboxes=obj.get("infoboxes") or {},
        raw=obj,
    )


def _load_local_tot(data_dir):
    data_dir = Path(data_dir)

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "dev" / "queries.jsonl"
    qrels_path = data_dir / "dev" / "qrels.txt"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

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
            query_text = _read_query_text(obj)

            queries[query_id] = query_text

    docs = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            obj = json.loads(line)
            doc_id, doc_obj = _make_doc(obj)
            docs[doc_id] = doc_obj

            if (i + 1) % 50000 == 0:
                print(f"[INFO] Loaded {i + 1} documents...")

    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()

            # Standard TREC qrels format:
            # query_id 0 doc_id relevance
            if len(parts) >= 4:
                query_id = str(parts[0])
                doc_id = str(parts[2])
                relevance = int(parts[3])

            # Fallback format:
            # query_id doc_id relevance
            elif len(parts) == 3:
                query_id = str(parts[0])
                doc_id = str(parts[1])
                relevance = int(parts[2])

            else:
                raise ValueError(f"Unexpected qrels line format: {line}")

            qrels.setdefault(query_id, {})[doc_id] = relevance

    print(
        f"[INFO] Loaded {len(queries)} queries, "
        f"{len(docs)} docs, "
        f"{len(qrels)} qrel query entries."
    )

    return queries, docs, qrels


def _load_irdatasets_tot(dataset_name):
    print(f"[INFO] Local data not found. Falling back to ir_datasets: {dataset_name}")

    import ir_datasets

    dataset = ir_datasets.load(dataset_name)

    queries = {}
    for query in dataset.queries_iter():
        query_id = str(query.query_id)
        query_text = getattr(query, "text", None) or getattr(query, "query", "")
        queries[query_id] = Query(query_id=query_id, text=str(query_text))

    docs = {}
    for doc in dataset.docs_iter():
        raw = {}

        for field in ["doc_id", "page_title", "title", "text", "sections", "infoboxes"]:
            if hasattr(doc, field):
                raw[field] = getattr(doc, field)

        doc_id = str(getattr(doc, "doc_id"))
        text = getattr(doc, "text", "")
        page_title = getattr(doc, "page_title", "") or getattr(doc, "title", "")
        sections = getattr(doc, "sections", {}) or {}
        infoboxes = getattr(doc, "infoboxes", {}) or {}

        parts = []
        if page_title:
            parts.append(str(page_title))
        if text:
            parts.append(str(text))
        _append_sections(parts, sections)
        _append_infoboxes(parts, infoboxes)

        docs[doc_id] = DocText(
            text=" ".join(parts),
            doc_id=doc_id,
            page_title=str(page_title or ""),
            sections=sections,
            infoboxes=infoboxes,
            raw=raw,
        )

    qrels = {}
    for qrel in dataset.qrels_iter():
        query_id = str(qrel.query_id)
        doc_id = str(qrel.doc_id)
        relevance = int(qrel.relevance)
        qrels.setdefault(query_id, {})[doc_id] = relevance

    print(
        f"[INFO] Loaded {len(queries)} queries, "
        f"{len(docs)} docs, "
        f"{len(qrels)} qrel query entries."
    )

    return queries, docs, qrels


def load_tot_dataset(dataset_name="trec-tot/2023/dev", data_dir="data"):
    """
    Load TREC-ToT dataset.

    Preferred local structure:

        data/
          corpus.jsonl
          dev/
            queries.jsonl
            qrels.txt

    If this structure exists, the loader uses local files.
    Otherwise, it falls back to ir_datasets.
    """

    data_dir = Path(data_dir)

    local_files_exist = (
        (data_dir / "corpus.jsonl").exists()
        and (data_dir / "dev" / "queries.jsonl").exists()
        and (data_dir / "dev" / "qrels.txt").exists()
    )

    if local_files_exist:
        return _load_local_tot(data_dir)

    return _load_irdatasets_tot(dataset_name)

