#!/usr/bin/env python3
"""
Router: the script Claude runs every turn.

Usage:
    python router.py "brief summary of user prompt"
    python router.py "brief summary" --top-k 3 --mode hybrid
    python router.py --session-start

Output:
    Prints the relevant instruction chunks with file provenance and
    confidence labels. Claude reads this output and follows the
    instructions for that turn.

The --session-start flag returns all chunks tagged session_start,
which are the instructions Claude needs at conversation init.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve paths relative to this script's location (the skill folder)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"

# Add src to path so imports work
sys.path.insert(0, str(SCRIPT_DIR))

from src.indexer import build_chunks
from src.query_parser import QueryParser
from src.retriever import HybridRetriever, Chunk


def format_result(chunk: "Chunk", score: float, rank: int,
                  confidence: str, method: str) -> str:
    """Format a single retrieval result for Claude to read."""
    lines = []
    lines.append(f"[{confidence.upper()}] (rank {rank}, score {score:.4f}, {method})")
    lines.append(f"  source: {chunk.file} L{chunk.line_start}–{chunk.line_end}")
    lines.append(f"  section: {chunk.section}")
    lines.append(f"  type: {chunk.instruction_type}")
    lines.append(f"  ---")
    # Indent the instruction text
    for line in chunk.text.split("\n"):
        lines.append(f"  {line}")
    lines.append("")
    return "\n".join(lines)


def format_session_start(chunks: list["Chunk"]) -> str:
    """Format all session_start chunks for conversation init."""
    lines = ["=== SESSION START INSTRUCTIONS ===", ""]
    for chunk in chunks:
        lines.append(f"## {chunk.section}")
        lines.append(f"   source: {chunk.file} L{chunk.line_start}–{chunk.line_end}")
        lines.append(f"   type: {chunk.instruction_type}")
        lines.append("")
        for line in chunk.text.split("\n"):
            lines.append(f"  {line}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve relevant system instructions for the current turn."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Brief summary of the user's prompt.",
    )
    parser.add_argument(
        "--session-start",
        action="store_true",
        help="Return all session_start instructions for conversation init.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return (default: 5).",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "tfidf", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "convex"],
        default="rrf",
        help="Fusion method when mode is hybrid (default: rrf).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum score threshold. Results below this are dropped.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="Root directory of the instruction repository.",
    )

    args = parser.parse_args()

    # Build chunks from the index + markdown files
    base_dir = args.base_dir
    chunks = build_chunks(base_dir)

    if not chunks:
        print("ERROR: No chunks built. Check index.json and markdown files.", file=sys.stderr)
        sys.exit(1)

    # Session start mode: return all session_start-tagged chunks
    if args.session_start:
        session_chunks = [c for c in chunks if "session_start" in c.tags]
        if args.json_output:
            records = [{
                "chunk_id": c.chunk_id,
                "file": c.file,
                "section": c.section,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "text": c.text,
                "instruction_type": c.instruction_type,
                "tags": c.tags,
            } for c in session_chunks]
            print(json.dumps({"mode": "session_start", "results": records}, indent=2))
        else:
            print(format_session_start(session_chunks))
        return

    # Normal retrieval mode: need a query
    if not args.query:
        print("ERROR: Provide a task summary or use --session-start.", file=sys.stderr)
        sys.exit(1)

    # Parse the query
    keywords_path = str(Path(base_dir) / "keywords.json")
    qp = QueryParser(keywords_path=keywords_path)
    parsed = qp.parse(args.query)

    # Build retriever and index
    retriever = HybridRetriever(
        mode=args.mode,
        fusion_method=args.fusion,
    )
    retriever.index(chunks)

    # Retrieve
    results = retriever.retrieve(
        query=parsed.expanded_query,
        top_k=args.top_k,
        required_tags=parsed.matched_tags if parsed.matched_tags else None,
    )

    # Apply threshold
    if args.threshold > 0:
        results = [r for r in results if r.score >= args.threshold]

    if not results:
        msg = f"No instructions matched for: {args.query}"
        if args.json_output:
            print(json.dumps({"mode": args.mode, "query": args.query, "results": [], "note": msg}))
        else:
            print(f"⚠ {msg}")
        return

    # Output
    if args.json_output:
        records = [{
            "rank": r.rank,
            "score": r.score,
            "confidence": r.confidence,
            "method": r.method,
            "chunk_id": r.chunk.chunk_id,
            "file": r.chunk.file,
            "section": r.chunk.section,
            "line_start": r.chunk.line_start,
            "line_end": r.chunk.line_end,
            "instruction_type": r.chunk.instruction_type,
            "tags": r.chunk.tags,
            "text": r.chunk.text,
        } for r in results]
        output = {
            "mode": args.mode,
            "fusion": args.fusion if args.mode == "hybrid" else None,
            "query": args.query,
            "parsed": {
                "verbs": parsed.verbs,
                "objects": parsed.objects,
                "constraints": parsed.constraints,
                "matched_tags": sorted(parsed.matched_tags),
            },
            "results": records,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"=== INSTRUCTIONS FOR: {args.query} ===")
        print(f"    mode: {args.mode} | fusion: {args.fusion} | top-k: {args.top_k}")
        print(f"    parsed tags: {sorted(parsed.matched_tags)}")
        print()
        for r in results:
            print(format_result(r.chunk, r.score, r.rank, r.confidence, r.method))


if __name__ == "__main__":
    main()
