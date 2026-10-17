"""
Extraction agent.

Converts each raw source's text into zero or more structured evidence
records (claim, excerpt, entity, topic). Uses the LLM per-source rather
than one giant call so a single bad/empty source can't derail the batch.

RAG note: rather than handing the LLM the first N truncated characters of
a page (which can miss the part of a long page that actually answers the
sub-question), long sources are chunked and the chunks most relevant to
the sub-question are retrieved via embedding similarity (see
utils/rag_store.py) and used as the context instead. Short sources go
through unchanged — retrieval only helps once there's more than one chunk
to choose between.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from utils import rag_store
from utils.llm_client import ask_json
from utils.models import ResearchJob, SourceRecord, EvidenceRecord

SYSTEM_PROMPT = """You are an evidence-extraction agent for a strategy research
pipeline. Given text retrieved from a web page and the research question it was
collected for, pull out only claims that are directly relevant and useful
to a strategy analyst. Do not invent facts that are not in the text. If the
page contains nothing relevant, return an empty list.
"""

_RAG_TOP_K_CHUNKS = 3  # how many chunks of a long source to keep as context
_MAX_DIRECT_CHARS = 6000  # pages up to this size go to the LLM whole
_WORKERS = 5


def _build_context(question: str, raw_text: str) -> str:
    """Chunk + retrieve the parts of `raw_text` most relevant to `question`."""
    if len(raw_text) <= _MAX_DIRECT_CHARS:
        return raw_text  # fits comfortably in one prompt: skip the (slow) embedding round-trip
    chunks = rag_store.chunk_text(raw_text)
    if len(chunks) <= _RAG_TOP_K_CHUNKS:
        return raw_text  # short enough that retrieval wouldn't trim anything meaningful
    top_chunks = rag_store.retrieve_top(question, chunks, top_k=_RAG_TOP_K_CHUNKS)
    return "\n---\n".join(top_chunks)


def _extract_from_source(question: str, source: SourceRecord) -> List[EvidenceRecord]:
    if not source.raw_text.strip():
        return []

    context = _build_context(question, source.raw_text)

    user_prompt = f"""
Research question this source was collected for: {question}
Source title: {source.title}
Source URL: {source.url}
Source text (most relevant excerpts, retrieved for this question):
\"\"\"{context}\"\"\"

Return a JSON array (can be empty) of evidence objects, each shaped as:
{{
  "claim": "<one-sentence factual claim, in your own words>",
  "excerpt": "<short supporting excerpt from the text, under 200 chars>",
  "entity": "<company/organization/market the claim is about>",
  "topic": "<short topic label>"
}}
Extract at most 4 of the strongest claims from this source.
"""
    data = ask_json(SYSTEM_PROMPT, user_prompt)
    if not isinstance(data, list):
        return []

    records = []
    for item in data:
        if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
            continue
        records.append(
            EvidenceRecord(
                source_id=source.id,
                task_id=source.task_id,
                claim=item.get("claim", ""),
                excerpt=item.get("excerpt", ""),
                entity=item.get("entity", ""),
                topic=item.get("topic", ""),
                url=source.url,
            )
        )
    return records


def extract_evidence(job: ResearchJob) -> List[EvidenceRecord]:
    if job.plan is None:
        raise RuntimeError("Cannot extract before a plan exists.")

    question_by_task = {t.id: t.question for t in job.plan.tasks}
    all_evidence: List[EvidenceRecord] = []

    def _work(source: SourceRecord):
        question = question_by_task.get(source.task_id, job.query)
        try:
            return source, _extract_from_source(question, source), None
        except Exception as e:  # noqa: BLE001
            return source, [], e

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for source, records, err in pool.map(_work, job.sources):
            if err:
                job.add_log(f"Extraction failed for {source.url}: {str(err)[:200]}")
            all_evidence.extend(records)

    job.evidence = all_evidence
    job.add_log(f"Extraction produced {len(all_evidence)} evidence records.")
    return all_evidence
