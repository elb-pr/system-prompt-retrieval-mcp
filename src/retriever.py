"""
Hybrid line-aware retriever: BM25 lexical backbone + optional TF-IDF semantic
reranker + RRF/convex fusion. Zero external dependencies beyond stdlib.

Design rationale (from IR literature):
- BM25 is the canonical lexical workhorse for terminology-dense corpora
- Hybrid retrieval improves recall when queries paraphrase instructions
- RRF is robust to score-scale differences between retrievers
- For small corpora (<1000 chunks), TF-IDF cosine is sufficient semantic signal
  without pulling sentence-transformers

References:
  BM25: Robertson & Zaragoza (2009)
  Hybrid fusion: Kuzi et al. (2020), arxiv:2010.01195
  RRF: Cormack et al. (2009)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ── Stopwords (compact set, English) ─────────────────────────────────────────

STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all each every both few more most other some such no nor "
    "not only own same so than too very and but if or because until "
    "while about against it its this that these those i me my we our "
    "you your he him his she her they them their what which who whom".split()
)


# ── Tokeniser ────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords and single chars."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single retrievable instruction chunk with line-level provenance."""
    chunk_id: str
    file: str
    section: str
    line_start: int
    line_end: int
    text: str
    summary: str
    keywords: list[str]
    tags: list[str]
    instruction_type: str = "general"

    # Populated at index time
    _token_freqs: Counter = field(default_factory=Counter, repr=False)
    _token_count: int = 0
    _tfidf_vector: dict[str, float] = field(default_factory=dict, repr=False)

    def indexable_text(self) -> str:
        """Concatenated searchable surface: text + summary + keywords."""
        parts = [self.text, self.summary, " ".join(self.keywords)]
        return " ".join(parts)


@dataclass
class RetrievalResult:
    """A scored chunk returned by the retriever."""
    chunk: Chunk
    score: float
    rank: int
    method: str  # "bm25", "tfidf", "hybrid"
    confidence: str  # "high", "medium", "low", "uncertain"


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25:
    """
    Okapi BM25 implementation over in-memory token frequency maps.

    Parameters:
        k1: Term frequency saturation. Default 1.5 (standard).
        b:  Length normalisation. Default 0.75 (standard).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[Chunk] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._n: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of Chunks."""
        self._docs = chunks
        self._n = len(chunks)

        # Tokenise and compute per-doc term frequencies
        total_tokens = 0
        df: Counter = Counter()  # document frequency per term

        for chunk in chunks:
            tokens = tokenise(chunk.indexable_text())
            chunk._token_freqs = Counter(tokens)
            chunk._token_count = len(tokens)
            total_tokens += len(tokens)

            # Each unique term in this doc increments df once
            for term in set(tokens):
                df[term] += 1

        self._avg_dl = total_tokens / max(self._n, 1)

        # IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log(
                (self._n - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """
        Score all documents against query tokens.
        Returns list of (doc_index, score) sorted descending by score.
        """
        scores = []
        for i, chunk in enumerate(self._docs):
            s = 0.0
            dl = chunk._token_count
            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = chunk._token_freqs.get(qt, 0)
                idf = self._idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (dl / self._avg_dl)
                )
                s += idf * (numerator / denominator)
            scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ── TF-IDF Semantic Similarity ───────────────────────────────────────────────

class TFIDFSemantic:
    """
    TF-IDF cosine similarity reranker. Lightweight semantic signal
    without ML model dependencies. Captures term importance weighting
    that raw BM25 keyword overlap misses (e.g. rare terms get higher weight).
    """

    def __init__(self):
        self._idf: dict[str, float] = {}
        self._docs: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Build TF-IDF vectors for all chunks."""
        self._docs = chunks
        n = len(chunks)

        # Document frequency
        df: Counter = Counter()
        for chunk in chunks:
            tokens = set(tokenise(chunk.indexable_text()))
            for t in tokens:
                df[t] += 1

        # IDF: log(N / df) + 1 (smoothed)
        self._idf = {
            term: math.log(n / freq) + 1.0 for term, freq in df.items()
        }

        # Build normalised TF-IDF vectors per chunk
        for chunk in chunks:
            tokens = tokenise(chunk.indexable_text())
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {}
            for term, count in tf.items():
                if term in self._idf:
                    vec[term] = (count / total) * self._idf[term]
            # L2 normalise
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            chunk._tfidf_vector = {k: v / norm for k, v in vec.items()}

    def score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """Cosine similarity between query TF-IDF vector and each chunk."""
        # Build query vector
        tf = Counter(query_tokens)
        total = len(query_tokens) or 1
        q_vec = {}
        for term, count in tf.items():
            if term in self._idf:
                q_vec[term] = (count / total) * self._idf[term]
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {k: v / q_norm for k, v in q_vec.items()}

        scores = []
        for i, chunk in enumerate(self._docs):
            dot = sum(
                q_vec.get(t, 0.0) * chunk._tfidf_vector.get(t, 0.0)
                for t in set(list(q_vec.keys()) + list(chunk._tfidf_vector.keys()))
            )
            scores.append((i, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ── Fusion ───────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).
    Combines multiple ranked lists by summing 1/(k + rank) per document.
    Robust to score-scale differences between retrievers.

    Args:
        ranked_lists: Each is a list of (doc_index, score) sorted desc.
        k: Smoothing constant. Default 60 (standard).

    Returns:
        Fused list of (doc_index, rrf_score) sorted descending.
    """
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def convex_fusion(
    lexical_scores: list[tuple[int, float]],
    semantic_scores: list[tuple[int, float]],
    alpha: float = 0.6,
) -> list[tuple[int, float]]:
    """
    Convex combination of normalised scores.
    final_score = alpha * norm_lexical + (1 - alpha) * norm_semantic

    Args:
        alpha: Weight for lexical scores. Default 0.6 (slight lexical bias
               appropriate for terminology-dense instruction corpora).
    """
    def normalise(scores: list[tuple[int, float]]) -> dict[int, float]:
        if not scores:
            return {}
        vals = [s for _, s in scores]
        min_s, max_s = min(vals), max(vals)
        rng = max_s - min_s if max_s > min_s else 1.0
        return {idx: (s - min_s) / rng for idx, s in scores}

    norm_lex = normalise(lexical_scores)
    norm_sem = normalise(semantic_scores)

    all_ids = set(norm_lex.keys()) | set(norm_sem.keys())
    combined = {}
    for idx in all_ids:
        combined[idx] = (
            alpha * norm_lex.get(idx, 0.0)
            + (1 - alpha) * norm_sem.get(idx, 0.0)
        )

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ── Confidence Thresholds ────────────────────────────────────────────────────

def classify_confidence(score: float, max_score: float) -> str:
    """
    Map a retrieval score to a confidence label.
    Thresholds are relative to the top score in the result set.
    """
    if max_score <= 0:
        return "uncertain"
    ratio = score / max_score
    if ratio >= 0.7:
        return "high"
    elif ratio >= 0.4:
        return "medium"
    elif ratio >= 0.15:
        return "low"
    return "uncertain"


# ── Tag Boosting ─────────────────────────────────────────────────────────────

def apply_tag_boost(
    scores: list[tuple[int, float]],
    chunks: list[Chunk],
    required_tags: Optional[set[str]] = None,
    boost_factor: float = 1.3,
) -> list[tuple[int, float]]:
    """
    Boost scores for chunks whose tags overlap with required_tags.
    This lets the keyword category → tag mapping from keywords.json
    influence ranking without replacing BM25 scoring.
    """
    if not required_tags:
        return scores

    boosted = []
    for idx, score in scores:
        chunk_tags = set(chunks[idx].tags)
        if chunk_tags & required_tags:
            overlap = len(chunk_tags & required_tags)
            score *= boost_factor ** overlap
        boosted.append((idx, score))

    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


# ── Main Retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates BM25 + optional TF-IDF semantic + fusion + tag boosting.

    Modes:
        "bm25"   - Lexical only (fast, precise term matching)
        "tfidf"  - Semantic only (TF-IDF cosine, catches paraphrases)
        "hybrid" - BM25 + TF-IDF fused via RRF (default, most robust)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        fusion_method: str = "rrf",  # "rrf" or "convex"
        alpha: float = 0.6,
        rrf_k: int = 60,
        tag_boost: float = 1.3,
    ):
        self.mode = mode
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.tag_boost = tag_boost

        self._bm25 = BM25()
        self._tfidf = TFIDFSemantic()
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Build all indexes from chunks."""
        self._chunks = chunks
        self._bm25.index(chunks)
        if self.mode in ("tfidf", "hybrid"):
            self._tfidf.index(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        required_tags: Optional[set[str]] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-k instruction chunks for a natural-language task summary.

        Args:
            query: Natural language task description.
            top_k: Number of results to return.
            required_tags: Optional tag filter to boost matching chunks.

        Returns:
            Ranked list of RetrievalResult with confidence labels.
        """
        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        if self.mode == "bm25":
            raw_scores = self._bm25.score(query_tokens)
            method = "bm25"
        elif self.mode == "tfidf":
            raw_scores = self._tfidf.score(query_tokens)
            method = "tfidf"
        else:  # hybrid
            bm25_scores = self._bm25.score(query_tokens)
            tfidf_scores = self._tfidf.score(query_tokens)

            if self.fusion_method == "rrf":
                raw_scores = reciprocal_rank_fusion(
                    bm25_scores, tfidf_scores, k=self.rrf_k
                )
            else:
                raw_scores = convex_fusion(
                    bm25_scores, tfidf_scores, alpha=self.alpha
                )
            method = "hybrid"

        # Tag boosting
        if required_tags:
            raw_scores = apply_tag_boost(
                raw_scores, self._chunks, required_tags, self.tag_boost
            )

        # Trim to top_k and build results
        top_scores = raw_scores[:top_k]
        if not top_scores:
            return []

        max_score = top_scores[0][1] if top_scores else 0.0

        results = []
        for rank, (idx, score) in enumerate(top_scores):
            if score <= 0:
                continue
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                score=round(score, 6),
                rank=rank + 1,
                method=method,
                confidence=classify_confidence(score, max_score),
            ))

        return results
