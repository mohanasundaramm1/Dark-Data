"""
P2: Retrieval Quality Evaluation Script
Runs a benchmark suite against the live search API and computes MRR, nDCG, and Precision@K.
Usage: PYTHONPATH=. rag_pipeline_env/bin/python src/evaluation/retrieval_eval.py
"""

import requests
import math
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrievalEval")

API_URL = "http://localhost:8888/search"

# ---------------------------------------------------------------------------
# Benchmark set: (query, list of keywords that MUST appear in top answers)
# Adjust these based on your actual ingested content.
# ---------------------------------------------------------------------------
BENCHMARK = [
    {"query": "leadership skills", "relevant_keywords": ["leader", "management", "decision"]},
    {"query": "data science fundamentals", "relevant_keywords": ["data", "analysis", "model", "statistics"]},
    {"query": "machine learning algorithms", "relevant_keywords": ["algorithm", "training", "model", "prediction"]},
    {"query": "project management methods", "relevant_keywords": ["project", "timeline", "scope", "stakeholder"]},
    {"query": "financial planning", "relevant_keywords": ["budget", "finance", "cost", "revenue", "investment"]},
]


def is_relevant(chunk_text: str, keywords: List[str]) -> bool:
    """A chunk is 'relevant' if it contains at least one of the expected keywords."""
    chunk_lower = chunk_text.lower()
    return any(kw.lower() in chunk_lower for kw in keywords)


def precision_at_k(results: List[Dict[str, Any]], relevant_keywords: List[str], k: int = 3) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for r in top_k if is_relevant(r["text"], relevant_keywords))
    return relevant_count / k


def reciprocal_rank(results: List[Dict[str, Any]], relevant_keywords: List[str]) -> float:
    """MRR: 1/rank for the first relevant result."""
    for i, r in enumerate(results):
        if is_relevant(r["text"], relevant_keywords):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(results: List[Dict[str, Any]], relevant_keywords: List[str], k: int = 3) -> float:
    """nDCG@K: normalized discounted cumulative gain at K."""
    dcg = 0.0
    for i, r in enumerate(results[:k]):
        rel = 1.0 if is_relevant(r["text"], relevant_keywords) else 0.0
        dcg += rel / math.log2(i + 2)
    # Ideal DCG: all top-k are relevant
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(results))))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(top_k: int = 5) -> Dict[str, float]:
    logger.info(f"Starting Retrieval Quality Evaluation against {API_URL}")
    all_mrr, all_p_at_3, all_ndcg = [], [], []

    for item in BENCHMARK:
        query = item["query"]
        keywords = item["relevant_keywords"]

        try:
            resp = requests.post(API_URL, json={"query": query, "top_k": top_k}, timeout=10)
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            logger.warning(f"Query failed: '{query}' -> {e}")
            all_mrr.append(0.0)
            all_p_at_3.append(0.0)
            all_ndcg.append(0.0)
            continue

        mrr = reciprocal_rank(results, keywords)
        p3 = precision_at_k(results, keywords, k=3)
        ndcg = ndcg_at_k(results, keywords, k=3)

        logger.info(f"Query: '{query}' | MRR={mrr:.3f} | P@3={p3:.3f} | nDCG@3={ndcg:.3f}")
        all_mrr.append(mrr)
        all_p_at_3.append(p3)
        all_ndcg.append(ndcg)

    metrics = {
        "MRR": round(sum(all_mrr) / len(all_mrr), 4),
        "Precision@3": round(sum(all_p_at_3) / len(all_p_at_3), 4),
        "nDCG@3": round(sum(all_ndcg) / len(all_ndcg), 4),
    }

    logger.info("=" * 50)
    logger.info("RETRIEVAL EVALUATION SUMMARY")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)
    return metrics


if __name__ == "__main__":
    run_evaluation()
