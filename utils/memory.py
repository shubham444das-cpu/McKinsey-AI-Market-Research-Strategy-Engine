"""
Knowledge/memory layer.

Public API (`remember_finding`, `recall_related`, `all_records`) is kept
stable so callers (aggregator, report_writer, the sidebar in app.py) don't
need to change. Internally this now delegates to `utils.rag_store`, which
chunks/embeds/retrieves by cosine similarity instead of the old raw
keyword-overlap match — see rag_store.py for the RAG implementation.
"""
from __future__ import annotations

from typing import Dict, List

from utils import rag_store


def remember_finding(topic: str, summary: str, source_urls: List[str]) -> None:
    """Store a validated, aggregated finding for future reuse."""
    rag_store.persist_finding(topic=topic, summary=summary, source_urls=source_urls)


def recall_related(query: str, limit: int = 5) -> List[Dict]:
    """
    Semantic recall of related prior findings via embedding similarity
    (see utils.rag_store.recall_similar). Strips the raw embedding vector
    before returning records, since callers only care about topic/summary/
    source_urls.
    """
    results = rag_store.recall_similar(query, limit=limit)
    return [{k: v for k, v in r.items() if k != "embedding"} for r in results]


def all_records() -> List[Dict]:
    return [{k: v for k, v in r.items() if k != "embedding"} for r in rag_store._load_persisted()]
