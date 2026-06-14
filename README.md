# Improving Neural Re-Ranking for Tip-of-the-Tongue Retrieval

This repository contains the code used for the IS584 term project **"Analyzing and Improving Tip-of-the-Tongue Retrieval with Neural Re-Ranking."** The project evaluates BM25, dense retrieval, dense-lexical hybrid retrieval, Reciprocal Rank Fusion, CrossEncoder re-ranking, rank-preserving fusion, and diagnostic error analysis on the TREC Tip-of-the-Tongue 2023 development set.

The main finding is that candidate generation and neural re-ranking fail in different ways: expanding and diversifying the candidate pool improves candidate availability, while unconstrained CrossEncoder re-ranking often over-promotes semantically plausible distractors.

---

## 1. Repository Structure

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── bm25_sweep.yaml
│   └── bm25_wandb_sweep.yaml
├── src/
│   ├── load_data.py
│   ├── metrics.py
│   ├── run_bm25.py
│   ├── run_bm25_sweep.py
│   ├── run_dense.py
│   ├── combine_rankings.py
│   ├── evaluate_candidate_recall.py
│   ├── compare_candidate_recall.py
│   ├── oracle_upper_bound.py
│   ├── run_reranker.py
│   ├── run_reranker_fusion.py
│   ├── run_gated_reranker.py
│   ├── run_reranker_rank_injected.py
│   ├── run_reranker_field_aware.py
│   ├── run_multiview_reranker_fusion.py
│   ├── statistical_test.py
│   ├── export_error_analysis.py
│   └── summarize_all_results.py
└── data/                  # Not included in the repository; create manually.

```
Generated files are written to `outputs/`. BM25 tokenization caches are written to `cache/`.

---

## 2. Environment Setup

Recommended Python version: **Python 3.10 or 3.11**.

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The experiments use `sentence-transformers` and PyTorch. If CUDA is available, the neural retrieval and re-ranking scripts automatically use GPU; otherwise, they run on CPU.

---

## 3. Dataset Setup

The code expects the TREC-ToT 2023 development data in the following local structure:

```text
data/
├── corpus.jsonl
└── dev/
    ├── queries.jsonl
    └── qrels.txt
```

The original `ir_datasets` download endpoint for TREC-ToT 2023 became unavailable during replication. Therefore, `src/load_data.py` first checks for the local `data/` directory above. If the files exist locally, they are used directly. If they do not exist, the loader falls back to `ir_datasets`.

After placing the files, verify the dataset can be loaded:

```bash
python -c "import sys; sys.path.append('src'); from load_data import load_tot_dataset; q,d,r = load_tot_dataset('trec-tot/2023/dev'); print(len(q), len(d), len(r))"
```

Expected output for the local TREC-ToT 2023 development setup:

```text
150 231848 150
```

---

## 4. Reproducibility and Random Seeds

All runnable experiment scripts explicitly set a fixed random seed of 42 at the beginning of execution. The seed block sets Python's `random`, NumPy, PyTorch, and CUDA seeds where available:

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Note: these experiments do not train models. Most computations are deterministic ranking, scoring, and evaluation steps. Minor floating-point differences may still occur across different hardware, CUDA versions, or PyTorch builds.

---

## 5. Step-by-Step Reproduction

Create output directories:

```bash
mkdir -p outputs cache reports/figures
```

### Step 1: Run the tuned BM25 baseline

```bash
python src/run_bm25.py \
  --dataset_name trec-tot/2023/dev \
  --k1 1.6 \
  --b 0.9 \
  --top_k 1000
```

This produces:

```text
outputs/rank_bm25_k1_1.6_b_0.9_topk_1000.json
outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json
```

Optional BM25 sweep:

```bash
python src/run_bm25_sweep.py
python src/summarize_bm25_results.py
```

### Step 2: Run dense retrieval

```bash
python src/run_dense.py \
  --dataset trec-tot/2023/dev \
  --model_name sentence-transformers/msmarco-MiniLM-L-6-v3 \
  --top_k 1000 \
  --batch_size 64 \
  --output_dir outputs
```

This produces:

```text
outputs/dense_sentence-transformers_msmarco-MiniLM-L-6-v3_topk_1000.json
outputs/dense_rankings_sentence-transformers_msmarco-MiniLM-L-6-v3_topk_1000.json
```

### Step 3: Build dense-lexical hybrid candidate pools

Hybrid union, using BM25@100 and Dense@1000:

```bash
python src/combine_rankings.py \
  --dataset trec-tot/2023/dev \
  --ranking_a outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --ranking_b outputs/dense_rankings_sentence-transformers_msmarco-MiniLM-L-6-v3_topk_1000.json \
  --label_a BM25 \
  --label_b Dense \
  --a_k 100 \
  --b_k 1000 \
  --method union \
  --max_output_k 1000 \
  --output_name hybrid_union_bm25_100_dense_1000
```

Hybrid Reciprocal Rank Fusion:

```bash
python src/combine_rankings.py \
  --dataset trec-tot/2023/dev \
  --ranking_a outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --ranking_b outputs/dense_rankings_sentence-transformers_msmarco-MiniLM-L-6-v3_topk_1000.json \
  --label_a BM25 \
  --label_b Dense \
  --a_k 100 \
  --b_k 1000 \
  --method rrf \
  --rrf_k 60 \
  --max_output_k 1000 \
  --output_name hybrid_rrf_bm25_100_dense_1000
```

### Step 4: Evaluate candidate availability

Candidate recall for one ranking file:

```bash
python src/evaluate_candidate_recall.py \
  --dataset_name trec-tot/2023/dev \
  --rankings_path outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json
```

McNemar comparison of candidate pools:

```bash
python src/compare_candidate_recall.py \
  --dataset trec-tot/2023/dev \
  --baseline_label BM25 \
  --baseline_rankings outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --baseline_k 100 \
  --system BM25=outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json Dense=outputs/dense_rankings_sentence-transformers_msmarco-MiniLM-L-6-v3_topk_1000.json Hybrid=outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --candidate_ks 100,500,1000 \
  --out_csv outputs/candidate_recall_comparison.csv \
  --out_json outputs/candidate_recall_comparison.json
```

Oracle upper-bound analysis:

```bash
python src/oracle_upper_bound.py \
  --dataset trec-tot/2023/dev \
  --rankings outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --candidate_ks 100,500,1000 \
  --eval_k 10 \
  --out_csv outputs/oracle_upper_bound.csv \
  --out_json outputs/oracle_upper_bound.json
```

### Step 5: Run pure CrossEncoder re-ranking

Pure CrossEncoder over BM25@1000:

```bash
python src/run_reranker.py \
  --dataset_name trec-tot/2023/dev \
  --bm25_rankings_path outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16
```

To preserve this output before running other pure CrossEncoder experiments, copy it:

```bash
cp outputs/reranker_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000.json outputs/reranker_bm25_1000_crossencoder_result.json
cp outputs/reranker_rankings_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000.json outputs/reranker_bm25_1000_crossencoder_rankings.json
```

Pure CrossEncoder over the hybrid union pool:

```bash
python src/run_reranker.py \
  --dataset_name trec-tot/2023/dev \
  --bm25_rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16
```

### Step 6: Run rank-preserving CrossEncoder fusion

```bash
python src/run_reranker_fusion.py \
  --dataset_name trec-tot/2023/dev \
  --bm25_rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --lambda_bm25 0.99
```

This produces:

```text
outputs/rank_fusion_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000_lambda_0.99.json
outputs/rank_fusion_rankings_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000_lambda_0.99.json
```

### Step 7: Run controlled re-ranking variants

Rank-injected CrossEncoder:

```bash
python src/run_reranker_rank_injected.py \
  --dataset trec-tot/2023/dev \
  --rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --output_dir outputs
```

Confidence-gated CrossEncoder:

```bash
python src/run_gated_reranker.py \
  --dataset trec-tot/2023/dev \
  --rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --max_promotions 3 \
  --min_ce_norm 0.95 \
  --min_gap 0.05 \
  --max_original_rank 1000 \
  --output_dir outputs
```

Field-aware CrossEncoder:

```bash
python src/run_reranker_field_aware.py \
  --dataset trec-tot/2023/dev \
  --rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --output_dir outputs
```

Multiview CrossEncoder fusion:

```bash
python src/run_multiview_reranker_fusion.py \
  --dataset trec-tot/2023/dev \
  --rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --lambda_rank 0.99 \
  --aggregation max \
  --output_dir outputs
```

Alternative BGE reranker can be tested by replacing the model name in fusion commands, for example:

```bash
python src/run_reranker_fusion.py \
  --dataset_name trec-tot/2023/dev \
  --bm25_rankings_path outputs/hybrid_union_bm25_100_dense_1000_rankings.json \
  --model_name BAAI/bge-reranker-base \
  --rerank_top_k 1000 \
  --batch_size 16 \
  --lambda_bm25 0.99
```

### Step 8: Run statistical testing

Compare BM25 against the final rank-preserving fusion system:

```bash
python src/statistical_test.py \
  --dataset_name trec-tot/2023/dev \
  --baseline_rankings outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --proposed_rankings outputs/rank_fusion_rankings_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000_lambda_0.99.json \
  --out_path outputs/statistical_test.json
```

### Step 9: Export rank-movement error analysis

```bash
python src/export_error_analysis.py \
  --bm25_rankings outputs/rank_bm25_rankings_k1_1.6_b_0.9_topk_1000.json \
  --ce_rankings outputs/reranker_bm25_1000_crossencoder_rankings.json \
  --fusion_rankings outputs/rank_fusion_rankings_cross-encoder_ms-marco-MiniLM-L-6-v2_topk_1000_lambda_0.99.json \
  --dataset_name trec-tot/2023/dev \
  --out_csv outputs/error_analysis_rank_movements.csv \
  --out_examples_csv outputs/error_analysis_representative_cases.csv
```

This produces:

```text
outputs/error_analysis_rank_movements.csv
outputs/error_analysis_representative_cases.csv
```

### Step 10: Summarize result files

```bash
python src/summarize_all_results.py
```

This produces:

```text
outputs/all_results_summary.csv
```

---

## 6. Main Scripts

| Script | Purpose |
|---|---|
| `src/load_data.py` | Loads local TREC-ToT files first, then falls back to `ir_datasets`. |
| `src/metrics.py` | Implements MRR@10, nDCG@10, and Recall@K. |
| `src/run_bm25.py` | Runs the BM25 lexical baseline. |
| `src/run_dense.py` | Runs dense retrieval with SentenceTransformers. |
| `src/combine_rankings.py` | Builds hybrid union or RRF candidate pools. |
| `src/run_reranker.py` | Runs pure CrossEncoder re-ranking. |
| `src/run_reranker_fusion.py` | Runs rank-preserving CrossEncoder fusion. |
| `src/run_gated_reranker.py` | Runs confidence-gated CrossEncoder re-ranking. |
| `src/run_reranker_rank_injected.py` | Runs CrossEncoder re-ranking with BM25 rank injected into the document text. |
| `src/run_reranker_field_aware.py` | Runs CrossEncoder re-ranking over compact field-aware document representations. |
| `src/run_multiview_reranker_fusion.py` | Scores multiple document views and applies rank-preserving fusion. |
| `src/statistical_test.py` | Runs query-level Wilcoxon significance testing. |
| `src/export_error_analysis.py` | Exports contrastive rank-movement and representative failure cases. |

---

## 7. Notes

- The local `data/` directory is not included in this repository.
- The first BM25 run creates a tokenized corpus cache under `cache/`, which speeds up later BM25 runs.
- W&B logging is optional. Do not use `--use_wandb` unless your W&B account is configured.
- The neural scripts download model weights from Hugging Face on first use.
- CPU execution is possible but substantially slower for dense retrieval and CrossEncoder re-ranking.

---

## 8. AI Use Statement

AI assistance was used for language editing in reports, README organization, and formatting support. All experiments, analysis decisions, and reported results were produced and verified by the author.

## 9. Report

Project Phase 3 report is under reports folder.

## 10. Note
Please reach me if there are any problems with executing the code.
