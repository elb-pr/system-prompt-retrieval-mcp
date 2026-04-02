"""
Indexer: reads the existing index.json and keyword categories, loads raw
markdown source files, and produces enriched Chunk objects ready for
the retriever to index.

The enriched schema per chunk:
  chunk_id, file, section, line_start, line_end, text, summary,
  keywords, tags, instruction_type

This aligns with the research-backed schema from the hybrid IR literature
(Pyserini JSON collections, SpecRover context retrieval, GACR query expansion).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .retriever import Chunk


def load_index(index_path: str) -> list[dict]:
    """Load the section definitions from index.json."""
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sections", [])


def load_keywords(keywords_path: str) -> dict:
    """Load keyword category definitions from keywords.json."""
    with open(keywords_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("categories", {})


def load_file_lines(file_path: str) -> list[str]:
    """Load a file and return lines (1-indexed conceptually, 0-indexed in list)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()


def extract_text(file_path: str, line_start: int, line_end: int) -> str:
    """
    Extract text from file between line_start and line_end (1-indexed, inclusive).
    Returns the raw text content of those lines.
    """
    lines = load_file_lines(file_path)
    # Convert to 0-indexed
    start = max(0, line_start - 1)
    end = min(len(lines), line_end)
    return "".join(lines[start:end]).strip()


def infer_instruction_type(tags: list[str], heading: str, text: str) -> str:
    """
    Classify the instruction type from tags and content.

    Types:
      procedure  - Step-by-step workflow or protocol
      rule       - Hard constraint or mandatory behaviour
      reference  - Background context, templates, notes
      meta       - Session/mode management, collaboration style
    """
    heading_lower = heading.lower()
    text_lower = text.lower()

    # Procedures: protocols, steps, workflows
    procedure_signals = [
        "step ", "protocol", "before claiming", "mandatory",
        "five-step", "at the start of", "after each", "sequence",
    ]
    if any(s in text_lower for s in procedure_signals):
        return "procedure"

    # Rules: constraints, forbidden, must/never
    rule_signals = [
        "forbidden", "must ", "never ", "always ", "only ",
        "do not", "cannot", "shall not",
    ]
    if any(s in text_lower for s in rule_signals):
        return "rule"

    # Meta: session management, collaboration, mode
    if "meta" in tags or "session_start" in tags:
        return "meta"

    # Templates and notes
    if "template" in heading_lower or "notes" in heading_lower:
        return "reference"

    return "general"


def extract_keywords_from_text(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract salient keywords from chunk text.
    Uses simple heuristic: words that appear capitalised or in bold/code,
    plus any terms from the keyword categories that appear in the text.
    """
    import re
    keywords = set()

    # Bold text: **word**
    bold = re.findall(r"\*\*([^*]+)\*\*", text)
    for b in bold:
        keywords.update(w.strip().lower() for w in b.split() if len(w.strip()) > 2)

    # Code spans: `word`
    code = re.findall(r"`([^`]+)`", text)
    for c in code:
        keywords.add(c.strip().lower())

    # Capitalised terms (likely proper nouns or emphasis)
    caps = re.findall(r"\b[A-Z][A-Z]+\b", text)
    for c in caps:
        if len(c) > 1:
            keywords.add(c.lower())

    return sorted(keywords)[:max_keywords]


def build_chunks(
    base_dir: str,
    index_path: Optional[str] = None,
    keywords_path: Optional[str] = None,
) -> list[Chunk]:
    """
    Build enriched Chunk objects from index.json + raw markdown files.

    Args:
        base_dir: Root directory of the instruction repository.
        index_path: Path to index.json. Defaults to base_dir/index.json.
        keywords_path: Path to keywords.json. Defaults to base_dir/keywords.json.

    Returns:
        List of Chunk objects ready for retriever indexing.
    """
    base = Path(base_dir)
    index_path = index_path or str(base / "index.json")
    keywords_path = keywords_path or str(base / "keywords.json")

    sections = load_index(index_path)
    _keyword_cats = load_keywords(keywords_path)

    # Build a reverse map: tag → all keywords that map to it
    tag_keywords: dict[str, set[str]] = {}
    for cat_name, cat_data in _keyword_cats.items():
        for tag in cat_data.get("maps_to_tags", []):
            if tag not in tag_keywords:
                tag_keywords[tag] = set()
            tag_keywords[tag].update(
                kw.lower() for kw in cat_data.get("keywords", [])
            )

    chunks: list[Chunk] = []

    for section in sections:
        file_rel = section["file"]
        file_path = str(base / file_rel)

        if not os.path.exists(file_path):
            continue

        line_start, line_end = section["lines"]
        text = extract_text(file_path, line_start, line_end)
        tags = section.get("tags", [])
        heading = section.get("heading", "")
        summary = section.get("description", "")

        # Extract keywords from text only — category keywords are kept
        # separate and influence retrieval via tag boosting in the query
        # parser, NOT by polluting the BM25 index. On a small corpus,
        # injecting 100+ shared category keywords drowns the per-chunk
        # lexical signal that BM25 relies on for discrimination.
        text_keywords = extract_keywords_from_text(text)
        all_keywords = sorted(set(text_keywords))

        instruction_type = infer_instruction_type(tags, heading, text)

        chunk = Chunk(
            chunk_id=section["id"],
            file=file_rel,
            section=heading,
            line_start=line_start,
            line_end=line_end,
            text=text,
            summary=summary,
            keywords=all_keywords,
            tags=tags,
            instruction_type=instruction_type,
        )
        chunks.append(chunk)

    return chunks


def export_enriched_index(chunks: list[Chunk], output_path: str) -> None:
    """Export enriched chunks to JSON for inspection/debugging."""
    records = []
    for c in chunks:
        records.append({
            "chunk_id": c.chunk_id,
            "file": c.file,
            "section": c.section,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "text": c.text,
            "summary": c.summary,
            "keywords": c.keywords,
            "tags": c.tags,
            "instruction_type": c.instruction_type,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"version": "2.0", "chunks": records}, f, indent=2)
