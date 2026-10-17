"""
Minimal "traditional" RAG (retrieval-augmented generation) layer:
chunk -> embed -> store -> retrieve -> stuff the retrieved text into an
LLM prompt. This replaces two things that used to work differently:

  1. `utils.memory` used to match past findings by raw keyword overlap.
     It now embeds and stores findings here, and recalls by cosine
     similarity — a real (if simple) vector-retrieval memory instead of a
     word-overlap heuristic.
  2. `agents.extractor` used to hand the LLM the first N truncated
     characters of a page. It now chunks the page, embeds the chunks, and
     retrieves only the chunks most relevant to the specific sub-question
     being researched — so a long page's relevant section isn't lost to
     truncation, and the extractor prompt stays small regardless of page
     length.
  3. `agents.aggregator` uses the same retrieval helper to select the most
     relevant accepted claims for the synthesis prompt when a Deep job
     produces more evidence than comfortably fits in one prompt, instead
     of naively dumping every claim in.

This is intentionally dependency-light: embeddings come from the Gemini
embedding endpoint (utils.llm_client.embed_texts), vectors are plain
Python lists, and similarity is brute-force cosine — fine at the scale of
one research job's evidence corpus. Swap this module for FAISS/Chroma/
pgvector later without touching callers; they only use `chunk_text`,
`rank_by_relevance`, `persist_finding`, and `recall_similar`.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List

from utils.llm_client import embed_texts

_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "rag_store.json")

_CHUNK_SIZE = 800     # characters per chunk
_CHUNK_OVERLAP = 120  # characters of overlap between consecutive chunks


# --------------------------------------------------------------- Chunking --

def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping fixed-size chunks for embedding/retrieval."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ------------------------------------------------------------- Similarity --

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_relevance(query: str, texts: List[str]) -> List[int]:
    """
    Return indices into `texts`, most-relevant-to-`query` first, by
    embedding cosine similarity. This is the core retrieval step: embed
    the query, embed the candidates, rank by similarity. Falls back to
    original order if embedding fails, so a transient embedding-API issue
    degrades to "use everything" rather than crashing the pipeline.
    """
    if not texts:
        return []
    try:
        vecs = embed_texts([query] + list(texts))
        query_vec, candidate_vecs = vecs[0], vecs[1:]
        if len(candidate_vecs) != len(texts):
            raise ValueError("embedding count mismatch")
    except Exception:
        return list(range(len(texts)))
    scored = sorted(range(len(texts)), key=lambda i: _cosine(query_vec, candidate_vecs[i]), reverse=True)
    return scored


def retrieve_top(query: str, texts: List[str], top_k: int) -> List[str]:
    """Convenience wrapper: return the top_k most relevant texts, ranked."""
    order = rank_by_relevance(query, texts)
    return [texts[i] for i in order[:top_k]]


# ------------------------------------------------------- Persistent memory --
# Cross-job "knowledge memory": same embed-and-compare mechanics, persisted
# to disk so a later research job can retrieve relevant past findings by
# meaning rather than by shared keywords.

def _load_persisted() -> List[Dict]:
    if not os.path.exists(_STORE_PATH):
        return []
    with open(_STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_persisted(records: List[Dict]) -> None:
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def persist_finding(topic: str, summary: str, source_urls: List[str]) -> None:
    """Embed and store a validated, aggregated finding for future recall."""
    try:
        vector = embed_texts([f"{topic}\n{summary}"])[0]
    except Exception:
        vector = []  # still store the finding even if embedding failed; it just won't be retrievable by similarity
    records = _load_persisted()
    records.append(
        {
            "topic": topic,
            "summary": summary,
            "source_urls": source_urls,
            "embedding": vector,
        }
    )
    _save_persisted(records)


def recall_similar(query: str, limit: int = 5, min_similarity: float = 0.3) -> List[Dict]:
    """Return past findings most semantically similar to `query`."""
    records = _load_persisted()
    embedded = [r for r in records if r.get("embedding")]
    if not embedded:
        return []
    try:
        query_vec = embed_texts([query])[0]
    except Exception:
        return list(reversed(embedded))[:limit]  # fallback: most recent
    scored = [(_cosine(query_vec, r["embedding"]), r) for r in embedded]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for score, r in scored[:limit] if score >= min_similarity]
