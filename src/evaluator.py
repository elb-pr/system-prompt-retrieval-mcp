"""
Evaluation harness for the hybrid retriever.

Metrics:
  - Top-1 file accuracy: Does the top result point to the correct file?
  - Top-1 chunk accuracy: Does the top result match the exact chunk_id?
  - Recall@k: Is the correct chunk in the top-k results?
  - Line-span IoU: Intersection-over-union of retrieved vs gold line spans
  - MRR: Mean Reciprocal Rank

The test set is a list of (task_summary, gold_chunk_id, gold_file, gold_lines)
tuples. BM25-only is always the baseline; any hybrid configuration must beat it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    """A labelled query with expected retrieval target."""
    query: str
    gold_chunk_id: str
    gold_file: str
    gold_line_start: int
    gold_line_end: int
    description: str = ""  # Human note on what this tests
    hard_negatives: list[str] | None = None  # Confusingly similar chunk_ids


@dataclass
class EvalResult:
    """Per-query evaluation result."""
    query: str
    gold_chunk_id: str
    top1_chunk_id: str
    top1_correct: bool
    top1_file_correct: bool
    rank_of_gold: int  # 0 if not found in top_k
    line_span_iou: float
    reciprocal_rank: float
    confidence: str
    score: float


def line_span_iou(
    retrieved_start: int, retrieved_end: int,
    gold_start: int, gold_end: int,
) -> float:
    """
    Intersection-over-union of two line spans.
    Both are 1-indexed, inclusive.
    """
    inter_start = max(retrieved_start, gold_start)
    inter_end = min(retrieved_end, gold_end)
    intersection = max(0, inter_end - inter_start + 1)

    union_start = min(retrieved_start, gold_start)
    union_end = max(retrieved_end, gold_end)
    union = max(1, union_end - union_start + 1)

    return intersection / union


def load_test_set(path: str) -> list[TestCase]:
    """Load test cases from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    for item in data.get("test_cases", []):
        cases.append(TestCase(
            query=item["query"],
            gold_chunk_id=item["gold_chunk_id"],
            gold_file=item["gold_file"],
            gold_line_start=item["gold_line_start"],
            gold_line_end=item["gold_line_end"],
            description=item.get("description", ""),
            hard_negatives=item.get("hard_negatives"),
        ))
    return cases


def evaluate(
    retriever,
    query_parser,
    test_cases: list[TestCase],
    top_k: int = 5,
) -> dict:
    """
    Run the evaluation suite.

    Args:
        retriever: HybridRetriever instance (already indexed).
        query_parser: QueryParser instance.
        test_cases: Labelled test queries.
        top_k: Number of results to retrieve per query.

    Returns:
        Dict with aggregate metrics and per-query results.
    """
    results: list[EvalResult] = []

    for tc in test_cases:
        parsed = query_parser.parse(tc.query)
        retrieved = retriever.retrieve(
            query=parsed.expanded_query,
            top_k=top_k,
            required_tags=parsed.matched_tags if parsed.matched_tags else None,
        )

        if not retrieved:
            results.append(EvalResult(
                query=tc.query,
                gold_chunk_id=tc.gold_chunk_id,
                top1_chunk_id="<none>",
                top1_correct=False,
                top1_file_correct=False,
                rank_of_gold=0,
                line_span_iou=0.0,
                reciprocal_rank=0.0,
                confidence="uncertain",
                score=0.0,
            ))
            continue

        top1 = retrieved[0]
        top1_correct = top1.chunk.chunk_id == tc.gold_chunk_id
        top1_file_correct = top1.chunk.file == tc.gold_file

        # Find rank of gold chunk
        rank_of_gold = 0
        gold_result = None
        for r in retrieved:
            if r.chunk.chunk_id == tc.gold_chunk_id:
                rank_of_gold = r.rank
                gold_result = r
                break

        # Line span IoU (against top-1 result, or gold if found)
        target = gold_result or top1
        iou = line_span_iou(
            target.chunk.line_start, target.chunk.line_end,
            tc.gold_line_start, tc.gold_line_end,
        )

        rr = 1.0 / rank_of_gold if rank_of_gold > 0 else 0.0

        results.append(EvalResult(
            query=tc.query,
            gold_chunk_id=tc.gold_chunk_id,
            top1_chunk_id=top1.chunk.chunk_id,
            top1_correct=top1_correct,
            top1_file_correct=top1_file_correct,
            rank_of_gold=rank_of_gold,
            line_span_iou=iou,
            reciprocal_rank=rr,
            confidence=top1.confidence,
            score=top1.score,
        ))

    # Aggregate metrics
    n = len(results)
    if n == 0:
        return {"error": "No test cases"}

    top1_chunk_acc = sum(1 for r in results if r.top1_correct) / n
    top1_file_acc = sum(1 for r in results if r.top1_file_correct) / n
    recall_at_k = sum(1 for r in results if r.rank_of_gold > 0) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    avg_iou = sum(r.line_span_iou for r in results) / n

    # Failure categorisation
    failures = {
        "wrong_file": [],
        "right_file_wrong_lines": [],
        "not_in_top_k": [],
    }
    for r in results:
        if r.rank_of_gold == 0:
            failures["not_in_top_k"].append(r.query)
        elif not r.top1_correct and r.top1_file_correct:
            failures["right_file_wrong_lines"].append(r.query)
        elif not r.top1_file_correct:
            failures["wrong_file"].append(r.query)

    return {
        "metrics": {
            "top1_chunk_accuracy": round(top1_chunk_acc, 4),
            "top1_file_accuracy": round(top1_file_acc, 4),
            f"recall_at_{top_k}": round(recall_at_k, 4),
            "mrr": round(mrr, 4),
            "avg_line_span_iou": round(avg_iou, 4),
            "total_queries": n,
        },
        "failures": {
            k: {"count": len(v), "queries": v}
            for k, v in failures.items()
        },
        "per_query": [
            {
                "query": r.query,
                "gold": r.gold_chunk_id,
                "top1": r.top1_chunk_id,
                "correct": r.top1_correct,
                "rank_of_gold": r.rank_of_gold,
                "iou": round(r.line_span_iou, 4),
                "rr": round(r.reciprocal_rank, 4),
                "confidence": r.confidence,
                "score": round(r.score, 6),
            }
            for r in results
        ],
    }


def run_comparison(
    base_dir: str,
    test_path: str,
    top_k: int = 5,
) -> dict:
    """
    Run evaluation across all retriever modes and compare.
    BM25 is the baseline; hybrid must beat it to justify complexity.
    """
    # Import here to avoid circular deps when used as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.indexer import build_chunks
    from src.retriever import HybridRetriever
    from src.query_parser import QueryParser

    chunks = build_chunks(base_dir)
    test_cases = load_test_set(test_path)
    keywords_path = os.path.join(base_dir, "keywords.json")
    parser = QueryParser(keywords_path=keywords_path)

    modes = ["bm25", "tfidf", "hybrid"]
    comparison = {}

    for mode in modes:
        retriever = HybridRetriever(mode=mode, fusion_method="rrf")
        retriever.index(chunks)
        result = evaluate(retriever, parser, test_cases, top_k=top_k)
        comparison[mode] = result["metrics"]
        if mode == "hybrid":
            comparison["hybrid_detail"] = result

    # Determine winner
    baseline = comparison["bm25"]["mrr"]
    for mode in ["tfidf", "hybrid"]:
        delta = comparison[mode]["mrr"] - baseline
        comparison[mode]["mrr_delta_vs_bm25"] = round(delta, 4)

    return comparison


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """Run evaluation from command line."""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent
    )
    test_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        base_dir, "test_queries.json"
    )

    print(f"Base dir: {base_dir}")
    print(f"Test set: {test_path}")
    print()

    results = run_comparison(base_dir, test_path)

    print("=" * 60)
    print("RETRIEVER COMPARISON")
    print("=" * 60)

    for mode in ["bm25", "tfidf", "hybrid"]:
        metrics = results[mode]
        print(f"\n{'─' * 40}")
        print(f"Mode: {mode.upper()}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Print failures from hybrid
    if "hybrid_detail" in results:
        detail = results["hybrid_detail"]
        print(f"\n{'─' * 40}")
        print("HYBRID FAILURES:")
        for category, data in detail["failures"].items():
            if data["count"] > 0:
                print(f"  {category}: {data['count']}")
                for q in data["queries"]:
                    print(f"    - {q}")

    print()


if __name__ == "__main__":
    main()
