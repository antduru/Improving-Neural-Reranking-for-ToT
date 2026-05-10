# Improving-Neural-Reranking-for-ToT

This project aims to improve ToT performance with adaptive query-aware neural re-ranking.

## EDA and Quality Check
- `src/eda.py` is the script to see stats of queries and documents, and the distributions. To run the code, 

## Implementation Scripts
- `src/load_data.py`: Loads the TREC Tip-of-the-Tongue dataset with queries, documents, and relevance judgments using `ir_datasets`.

- `src/metrics.py`: Implements the evaluation metrics used in the project, including MRR@10, nDCG@10, and Recall@k.

- `src/run_bm25.py`: Runs the BM25 lexical baseline with selected `k1`, `b`, and `top_k` parameters, then saves rankings and evaluation results.

- `src/run_bm25_sweep.py`: Executes a small scripted BM25 parameter sweep over selected `k1` and `b` configurations.

- `src/log_bm25_wandb_sweep.py`: Logs saved BM25 sweep results into a W&B sweep without recomputing the retrieval runs.

- `src/summarize_bm25_results.py`: Aggregates BM25 result JSON files into a single CSV summary table.

- `src/run_reranker.py`: Applies a cross-encoder re-ranker to BM25 candidates and evaluates the re-ranked output.

- `src/run_reranker_fusion.py`: Combines BM25 rank or score signals with cross-encoder scores using weighted fusion.

- `src/evaluate_candidate_recall.py`: Computes candidate recall at different pool sizes such as top-100, top-500, and top-1000.

- `src/summarize_all_results.py`: Aggregates BM25, cross-encoder, and fusion experiment results into one summary CSV.

- `src/statistical_test.py`: Performs query-level Wilcoxon signed-rank testing between the BM25 baseline and the proposed fusion method.

python src/eda.py

Package Versions:
ir-datasets==0.5.11
rank-bm25==0.2.2
wandb==0.26.1
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
tqdm==4.65.0
sentence-transformers==5.4.1
torch==2.4.1
scikit-learn==1.4.2
matplotlib==3.8.4
pyyaml==6.0.2

AI USE STATEMENT:
In this project, AI is used for code formatting (not code generation), and ReadMe generation.
