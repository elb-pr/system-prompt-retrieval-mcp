"""
Query parser: decomposes a natural-language task summary into structured
query components for the retriever.

Extracts:
  - verbs (actions: compare, score, extract, verify, write)
  - objects (targets: GitHub repos, JSON index, task board)
  - constraints (qualifiers: single file, exact lines, python)
  - inferred tags (from keyword category matching)

Based on NL→code retrieval patterns (CodeSearchNet, GACR) adapted
for NL→instruction retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ParsedQuery:
    """Structured representation of a task summary for retrieval."""
    raw: str
    tokens: list[str]
    verbs: list[str]
    objects: list[str]
    constraints: list[str]
    matched_tags: set[str]
    expanded_query: str  # The final query string sent to the retriever

    def __repr__(self) -> str:
        return (
            f"ParsedQuery(verbs={self.verbs}, objects={self.objects}, "
            f"constraints={self.constraints}, tags={self.matched_tags})"
        )


# ── Verb/object lexicons ─────────────────────────────────────────────────────

ACTION_VERBS = {
    "verify", "check", "validate", "test", "confirm", "assert",
    "create", "build", "generate", "write", "produce", "draft",
    "edit", "modify", "update", "change", "fix", "patch", "refactor",
    "read", "extract", "parse", "inspect", "examine", "review",
    "plan", "design", "architect", "structure", "organise", "organize",
    "deploy", "push", "publish", "release", "ship",
    "compare", "evaluate", "score", "rank", "benchmark",
    "search", "find", "retrieve", "look up", "query",
    "log", "record", "track", "document", "note",
    "start", "begin", "init", "bootstrap", "setup",
    "complete", "finish", "close", "done", "wrap up",
    "escalate", "surface", "flag", "warn", "alert",
    "diagram", "visualise", "visualize", "chart", "draw",
}

OBJECT_NOUNS = {
    "file", "files", "document", "documents", "code", "script",
    "task", "tasks", "plan", "plans", "decision", "decisions",
    "skill", "skills", "tool", "tools", "mcp", "server",
    "repo", "repository", "github", "git",
    "test", "tests", "suite", "spec",
    "prompt", "instruction", "instructions", "system prompt",
    "notes", "log", "session", "context",
    "diagram", "chart", "output", "artifact", "artefact",
    "linear", "ticktick", "board",
    "error", "bug", "issue", "warning",
}


# ── Tag inference from keywords.json ─────────────────────────────────────────

class QueryParser:
    """Parses task summaries into structured queries with tag inference."""

    def __init__(self, keywords_path: Optional[str] = None):
        self._categories: dict = {}
        if keywords_path and Path(keywords_path).exists():
            with open(keywords_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._categories = data.get("categories", {})

    def parse(self, task_summary: str) -> ParsedQuery:
        """
        Parse a task summary into a structured query.

        Strategy:
          1. Tokenise the summary
          2. Identify action verbs and object nouns
          3. Match against keyword categories to infer tags
          4. Build the expanded query string
        """
        raw = task_summary.strip()
        # Basic tokenisation (preserve multi-word terms)
        tokens = re.findall(r"[a-z0-9_./-]+", raw.lower())

        # Extract verbs
        verbs = [t for t in tokens if t in ACTION_VERBS]

        # Extract objects
        objects = [t for t in tokens if t in OBJECT_NOUNS]

        # Constraints: anything remaining that's not a stopword or common word
        from .retriever import STOPWORDS
        constraints = [
            t for t in tokens
            if t not in ACTION_VERBS
            and t not in OBJECT_NOUNS
            and t not in STOPWORDS
            and len(t) > 2
        ]

        # Tag inference: match tokens against keyword categories
        matched_tags: set[str] = set()
        tokens_set = set(tokens)
        # Also check bigrams for multi-word keywords
        bigrams = set()
        for i in range(len(tokens) - 1):
            bigrams.add(f"{tokens[i]} {tokens[i+1]}")

        for _cat_name, cat_data in self._categories.items():
            cat_keywords = {kw.lower() for kw in cat_data.get("keywords", [])}
            # Check single tokens and bigrams against category keywords
            if tokens_set & cat_keywords or bigrams & cat_keywords:
                for tag in cat_data.get("maps_to_tags", []):
                    matched_tags.add(tag)

        # Build expanded query: verbs + objects + constraints + raw summary
        # The raw summary goes in because BM25 benefits from the full surface
        expanded_parts = verbs + objects + constraints
        expanded_query = raw  # Use full summary as query (BM25 handles weighting)

        return ParsedQuery(
            raw=raw,
            tokens=tokens,
            verbs=verbs,
            objects=objects,
            constraints=constraints,
            matched_tags=matched_tags,
            expanded_query=expanded_query,
        )
