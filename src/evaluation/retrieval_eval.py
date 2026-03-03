"""
P2 — Retrieval Quality Evaluation
==================================
Computes MRR@K, Precision@K, nDCG@K, and Recall@K against a labelled
benchmark of (query → expected_chunk_ids).

Modes
-----
* Label mode (accurate): supply known chunk_id values in BENCHMARK below.
* Keyword fallback mode: when expected_chunk_ids is empty, falls back to
  keyword heuristics so the script still produces useful smoke-check numbers
  even without labelled data.

Usage
-----
    PYTHONPATH=. rag_pipeline_env/bin/python src/evaluation/retrieval_eval.py
    # or
    make eval
"""

import math
import logging
import requests
from typing import Any, Dict, List, Sequence, Set, TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrievalEval")

API_URL = "http://localhost:8888/search"
EVAL_TOP_K = 5


# =============================================================================
# Benchmark definition
# =============================================================================

class BenchmarkItem(TypedDict):
    query: str
    # Preferred: real chunk_id UUIDs returned by the /search endpoint after ingestion.
    # Leave empty to fall back to keyword heuristics.
    expected_chunk_ids: List[str]
    # Fallback: keywords that should appear in relevant results.
    relevant_keywords: List[str]


BENCHMARK: List[BenchmarkItem] = [
    {
        "query": "leadership skills",
        "expected_chunk_ids": [],
        "relevant_keywords": ["leader", "management", "decision"],
    },
    {
        "query": "data science fundamentals",
        "expected_chunk_ids": [],
        "relevant_keywords": ["data", "analysis", "model", "statistics"],
    },
    {
        "query": "machine learning algorithms",
        "expected_chunk_ids": [],
        "relevant_keywords": ["algorithm", "training", "model", "prediction"],
    },
    {
        "query": "project management methods",
        "expected_chunk_ids": [],
        "relevant_keywords": ["project", "timeline", "scope", "stakeholder"],
    },
    {
        "query": "financial planning and budgeting",
        "expected_chunk_ids": [],
        "relevant_keywords": ["budget", "finance", "cost", "revenue", "investment"],
    },
]


# =============================================================================
# Relevance judgement
# =============================================================================

def _is_relevant_by_id(chunk_id: str, expected_ids: Set[str]) -> bool:
    return chunk_id in expected_ids


def _is_relevant_by_keyword(text: str, keywords: Sequence[str]) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def _is_relevant(result: Dict[str, Any], item: BenchmarkItem) -> bool:
    if item["expected_chunk_ids"]:
        return _is_relevant_by_id(result["chunk_id"], set(item["expected_chunk_ids"]))
    return _is_relevant_by_keyword(result["text"], item["relevant_keywords"])


# =============================================================================
# Metrics
# =============================================================================

def precision_at_k(results: List[Dict[str, Any]], item: BenchmarkItem, k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if _is_relevant(r, item))
    return hits / k


def recall_at_k(results: List[Dict[str, Any]], item: BenchmarkItem, k: int) -> float:
    n_relevant = len(item["expected_chunk_ids"]) or 1  # fallback: treat 1 as denom
    hits = sum(1 for r in results[:k] if _is_relevant(r, item))
    return hits / n_relevant


def reciprocal_rank(results: List[Dict[str, Any]], item: BenchmarkItem) -> float:
    for i, r in enumerate(results):
        if _is_relevant(r, item):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results: List[Dict[str, Any]], item: BenchmarkItem, k: int) -> float:
    dcg = sum(
        (1.0 if _is_relevant(r, item) else 0.0) / math.log2(i + 2)
        for i, r in enumerate(results[:k])
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, max(len(item["expected_chunk_ids"]), 1))))
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Runner
# =============================================================================

def run_evaluation(top_k: int = EVAL_TOP_K, api_url: str = API_URL) -> Dict[str, float]:
    logger.info(f"Retrieval Evaluation  →  {api_url}  (top_k={top_k})")
    mode = "label" if any(b["expected_chunk_ids"] for b in BENCHMARK) else "keyword-heuristic"
    logger.info(f"Relevance mode: {mode}")

    all_mrr: List[float] = []
    all_p3: List[float] = []
    all_ndcg: List[float] = []
    all_r5: List[float] = []

    for item in BENCHMARK:
        query = item["query"]
        try:
            resp = requests.post(api_url, json={"query": query, "top_k": top_k}, timeout=10)
            resp.raise_for_status()
            results: List[Dict[str, Any]] = resp.json()
        except Exception as exc:
            logger.warning(f"Query failed: '{query}'  →  {exc}")
            all_mrr.append(0.0)
            all_p3.append(0.0)
            all_ndcg.append(0.0)
            all_r5.append(0.0)
            continue

        mrr   = reciprocal_rank(results, item)
        p3    = precision_at_k(results, item, k=3)
        ndcg  = ndcg_at_k(results, item, k=3)
        r5    = recall_at_k(results, item, k=5)

        logger.info(
            f"  [{query[:35]:<35}]  MRR={mrr:.3f}  P@3={p3:.3f}  nDCG@3={ndcg:.3f}  R@5={r5:.3f}"
        )
        all_mrr.append(mrr)
        all_p3.append(p3)
        all_ndcg.append(ndcg)
        all_r5.append(r5)

    n = len(BENCHMARK)
    metrics = {
        "MRR":          round(sum(all_mrr)  / n, 4),
        "Precision@3":  round(sum(all_p3)   / n, 4),
        "nDCG@3":       round(sum(all_ndcg) / n, 4),
        "Recall@5":     round(sum(all_r5)   / n, 4),
    }

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v}")
    logger.info("=" * 60)

    if mode == "keyword-heuristic":
        logger.info(
            "TIP: Populate 'expected_chunk_ids' in BENCHMARK for accurate, "
            "label-based scoring instead of keyword heuristics."
        )

    return metrics


if __name__ == "__main__":
    run_evaluation()
