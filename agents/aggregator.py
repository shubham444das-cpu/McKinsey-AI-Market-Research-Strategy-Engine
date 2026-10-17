"""
Aggregation agent.

Groups accepted evidence into a handful of strategic themes (market shifts,
competitor moves, risks/opportunities) so the report writer has structure
to work from rather than a flat list of claims. Also pulls in any related
prior findings from memory.

RAG note: for Deep-depth jobs the accepted-evidence list can get large
enough that dumping every claim into one prompt wastes budget on marginal
items. When that happens, `utils.rag_store.rank_by_relevance` retrieves the
claims most relevant to the research objective (embedding similarity) and
only those go into the synthesis prompt. Nothing is silently dropped from
the job, though — the report writer's evidence appendix always lists every
accepted (and excluded) record regardless of what made it into this prompt.
"""
from __future__ import annotations

import json
from typing import List

from utils import rag_store
from utils.llm_client import ask_json
from utils.memory import recall_related
from utils.models import ResearchJob, ThemeGroup

SYSTEM_PROMPT = """You are a strategy synthesis agent. Group a list of
validated research claims into a small number of coherent themes useful for
a client strategy brief (e.g. market growth signals, competitive moves,
regulatory/risk factors, emerging opportunities). Every claim id must be
assigned to exactly one theme.
"""

_MAX_CLAIMS_PER_PROMPT = 40  # above this, retrieve the most relevant subset instead of dumping everything


def aggregate(job: ResearchJob) -> List[ThemeGroup]:
    accepted = [e for e in job.evidence if e.accepted]
    if not accepted:
        job.themes = []
        job.add_log("Aggregation skipped: no accepted evidence to group.")
        return []

    objective = job.plan.objective if job.plan else job.query

    claims_for_prompt = accepted
    if len(accepted) > _MAX_CLAIMS_PER_PROMPT:
        order = rag_store.rank_by_relevance(objective, [e.claim for e in accepted])
        claims_for_prompt = [accepted[i] for i in order[:_MAX_CLAIMS_PER_PROMPT]]
        job.add_log(
            f"Aggregation: {len(accepted)} accepted claims exceeded the prompt budget; "
            f"RAG-retrieved the {_MAX_CLAIMS_PER_PROMPT} most relevant to the objective."
        )

    items_payload = [
        {"id": e.id, "claim": e.claim, "entity": e.entity, "topic": e.topic}
        for e in claims_for_prompt
    ]
    user_prompt = f"""
Research objective: {objective}

Validated claims:
{json.dumps(items_payload, ensure_ascii=False)}

Return a JSON array of theme groups:
[
  {{
    "theme": "<short theme name>",
    "evidence_ids": ["<id>", "..."],
    "summary": "<2-3 sentence synthesis of what this theme means for the client>"
  }}
]
"""
    try:
        data = ask_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
    except Exception as e:  # noqa: BLE001
        job.add_log(f"Aggregation LLM call failed ({e}); using a single catch-all theme.")
        data = [{"theme": "All validated findings", "evidence_ids": [x.id for x in accepted],
                 "summary": " ".join(x.claim for x in accepted[:6])}]
    if isinstance(data, dict):
        data = data.get("themes", [])

    themes = []
    for t in data if isinstance(data, list) else []:
        if not isinstance(t, dict):
            continue
        themes.append(
            ThemeGroup(
                theme=t.get("theme", ""),
                evidence_ids=t.get("evidence_ids", []),
                summary=t.get("summary", ""),
            )
        )

    job.themes = themes
    job.add_log(f"Aggregation grouped evidence into {len(themes)} themes.")

    # Surface related prior research from memory (RAG recall, per dossier
    # section 5.2) and keep it on the job so the UI can display it.
    related = recall_related(job.query)
    job.related_memory = related
    if related:
        job.add_log(f"RAG memory recall found {len(related)} related prior finding(s).")

    return themes


def related_memory(job: ResearchJob):
    return recall_related(job.query)
