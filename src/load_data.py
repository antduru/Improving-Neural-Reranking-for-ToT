import ir_datasets

def load_tot_dataset(dataset_name: str = "trec-tot/2023/dev"):
    dataset = ir_datasets.load(dataset_name)

    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    docs = {}
    for doc in dataset.docs_iter():
        text_parts = []

        for attr in doc._fields:
            if attr == "doc_id":
                continue

            value = getattr(doc, attr)

            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                value = " ".join(map(str, value))

            text_parts.append(str(value))

        docs[doc.doc_id] = " ".join(text_parts)

    qrels_dict = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels_dict.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    return queries, docs, qrels_dict


if __name__ == "__main__":
    queries, docs, qrels = load_tot_dataset()

    print(f"Loaded {len(queries)} queries, {len(docs)} documents, and {sum(len(v) for v in qrels.values())} relevant pairs.")

    first_qid = next(iter(queries))
    print()
    print(f"Example query {first_qid}:")
    print(queries[first_qid])

    first_rel_doc_id = next(iter(qrels[first_qid]))
    print()
    print(f"Example relevant docs for query {first_qid}: {qrels[first_qid]}")
    print()
    print(f"Document {first_rel_doc_id}:")
    print(docs[first_rel_doc_id][:1000])