# NAME

## Fusion Reranker

![fusion.py](https://github.com/hojat-gazestani/machine-learning-lab/blob/main/rag-systems/retrieval/fusion.py)

**Reciprocal Rank Fusion (RRF)**

A **rank aggregation algorithm** used to merge multiple ranked retrieval lists into a single stronger ranking.

It merges multiple ranked retrieval results into one stronger ranking using only positions (ranks), not scores.

```text
Query
  ↓
Multi-query generator
  ↓
Retriever (multiple queries)
  ↓
RRF (fusion)   ← YOU ARE HERE
  ↓
Prompt
  ↓
LLM
```

The RRF is a bridge between retrieval and generation
