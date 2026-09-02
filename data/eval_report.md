# 📊 Enterprise RAG Evaluation & Benchmark Report

**Total Test Queries:** 10 | **Top-K:** 5 | **Evaluation Duration:** 518.93s

## 1. Retrieval Performance Scorecard

| Retrieval Metric | Baseline (Dense + BM25) | Advanced (With Neural Reranker) | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall@5** | 100.0% | 100.0% | +0.0% |
| **Precision@5** | 76.0% | 82.0% | +6.0% |
| **MRR (Mean Reciprocal Rank)** | 1.000 | 1.000 | +0.000 |
| **NDCG@5** | 1.356 | 1.483 | +0.127 |

## 2. Generation & Grounding Quality (LLM-as-a-Judge)

| Generation Metric | Score | Description |
| :--- | :--- | :--- |
| **Faithfulness** | **89.5%** | Measure of zero hallucination — claims are strictly grounded in retrieved evidence. |
| **Context Relevance** | **93.0%** | Proportion of retrieved chunks containing relevant answer context. |
| **Answer Relevance** | **91.0%** | Directness, accuracy, and completeness in answering user query. |
| **Citation Accuracy** | **92.8%** | Validity and accuracy of inline citations mapped to source chunks. |

## 3. Detailed Per-Query Results

| ID | Category | Role | Recall@5 | MRR | Faithfulness | Citation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `eval_001` | HR Policies | `hr` | 100% | 1.00 | 95% | `VERIFIED` |
| `eval_002` | HR Policies | `hr` | 100% | 1.00 | 95% | `VERIFIED` |
| `eval_003` | HR Policies | `hr` | 100% | 1.00 | 85% | `VERIFIED` |
| `eval_004` | Engineering & Careers | `engineer` | 100% | 1.00 | 85% | `VERIFIED` |
| `eval_005` | Incidents & Postmortems | `engineer` | 100% | 1.00 | 95% | `VERIFIED` |
| `eval_006` | Incidents & Postmortems | `engineer` | 100% | 1.00 | 95% | `VERIFIED` |
| `eval_007` | Finance & Legal | `finance` | 100% | 1.00 | 85% | `VERIFIED` |
| `eval_008` | Finance & Legal | `legal` | 100% | 1.00 | 90% | `VERIFIED` |
| `eval_009` | HR & Compensation | `hr` | 100% | 1.00 | 85% | `VERIFIED` |
| `eval_010` | General & Culture | `hr` | 100% | 1.00 | 85% | `VERIFIED` |
