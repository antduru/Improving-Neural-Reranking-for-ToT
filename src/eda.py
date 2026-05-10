import ir_datasets
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Load datasets
docs_ds = ir_datasets.load("trec-tot/2024")
queries_ds = ir_datasets.load("trec-tot/2023/dev")

# get queries and qrels
queries = [q.text for q in queries_ds.queries_iter()]
qrels = list(queries_ds.qrels_iter())
print(qrels[0])

query_lengths = [len(q.split()) for q in queries]

print("Num queries:", len(queries))
print("Avg query length:", sum(query_lengths) / len(query_lengths))

# Sample documents and analyze document amount and length
docs = []
for i, d in enumerate(docs_ds.docs_iter()):
    if i > 50000:  # sample first 50k
        break
    docs.append(d.text)

doc_lengths = [len(d.split()) for d in docs]

print("Num docs (sampled):", len(docs))
print("Avg doc length:", sum(doc_lengths) / len(doc_lengths))


# Analyze query-document overlap
def overlap(q, d):
    q_set = set(q.split())
    d_set = set(d.split())
    return len(q_set & d_set) / len(q_set) if len(q_set) > 0 else 0


docs_list = docs
overlaps = []

for q in queries:
    d = random.choice(docs_list)
    overlaps.append(overlap(q, d))

print("Avg overlap:", sum(overlaps) / len(overlaps))

# Analyze qrels - document - query relationships
qrel_counts = defaultdict(int)

for q in qrels:
    qrel_counts[q.query_id] += 1

counts = list(qrel_counts.values())

print("Avg relevant docs per query:", sum(counts) / len(counts))
print("Min relevant:", min(counts))
print("Max relevant:", max(counts))


empty_queries = sum(1 for q in queries if len(q.strip()) == 0)
print("Empty queries:", empty_queries)

# Plot query length distribution
plt.hist(query_lengths, bins=30)
plt.title("Query Length Distribution")
plt.xlabel("Query Length")
plt.ylabel("Count")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
out_path = os.path.join(
    repo_root, "reports", "figures", "query_length_distribution.png"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()
