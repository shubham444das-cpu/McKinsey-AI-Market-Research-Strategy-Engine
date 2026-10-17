"""
Report writer agent.

Produces the final structured brief using ONLY validated/accepted evidence
and the aggregated themes. Also writes a memory entry so future related
queries can reuse this synthesis (dossier section 5.2, Knowledge & Memory
Workspace).
"""
from __future__ import annotations

from utils.llm_client import ask_json
from utils.memory import remember_finding
from utils.models import ResearchJob, Report

SYSTEM_PROMPT = """You are a McKinsey-style strategy report writer. Using
ONLY the provided validated evidence and themes, write a structured research
brief. Never state a claim that isn't traceable to the provided evidence.
Where evidence is thin, say so explicitly rather than filling gaps with
generic industry knowledge.
"""


def _build_evidence_appendix_md(job: ResearchJob) -> str:
    lines = ["| Claim | Entity | Confidence | Source | Flags |", "|---|---|---|---|---|"]
    for e in job.evidence:
        flags = ", ".join(e.flags) if e.flags else "—"
        status = "accepted" if e.accepted else "excluded"
        lines.append(
            f"| {e.claim} | {e.entity} | {e.confidence:.2f} ({status}) | "
            f"[{e.url}]({e.url}) | {flags} |"
        )
    return "\n".join(lines)


def write_report(job: ResearchJob) -> Report:
    themes_payload = [
        {"theme": t.theme, "summary": t.summary, "evidence_count": len(t.evidence_ids)}
        for t in job.themes
    ]
    accepted_count = sum(1 for e in job.evidence if e.accepted)

    user_prompt = f"""
Client research request: {job.query}
Market/sector: {job.market or "not specified"}
Geography: {job.geography or "not specified"}
Competitors named: {job.competitors or "none specified"}

Themes and synthesis:
{themes_payload}

Total validated evidence items used: {accepted_count}

Return a JSON object with these keys, each a string of well-formed
Markdown-ready prose (no further JSON nesting):
{{
  "executive_summary": "...",
  "key_findings": "...",
  "market_signals": "...",
  "competitor_observations": "...",
  "implications": "...",
  "recommendations": "..."
}}
Keep each section tight (3-6 sentences or a short bullet list). If a section
has no supporting evidence, say so plainly instead of inventing content.
"""
    data = ask_json(SYSTEM_PROMPT, user_prompt, max_tokens=2200)
    if not isinstance(data, dict):
        data = {}

    def _s(key: str) -> str:
        v = data.get(key, "")
        if isinstance(v, list):
            return "\n".join(f"- {x}" for x in v)
        return str(v or "")

    report = Report(
        executive_summary=_s("executive_summary"),
        key_findings=_s("key_findings"),
        market_signals=_s("market_signals"),
        competitor_observations=_s("competitor_observations"),
        implications=_s("implications"),
        recommendations=_s("recommendations"),
        evidence_appendix_md=_build_evidence_appendix_md(job),
    )
    job.report = report
    job.add_log("Report generation complete.")

    # Store a compact memory entry for future related queries
    if job.themes:
        summary = " ".join(t.summary for t in job.themes)[:800]
        urls = list({e.url for e in job.evidence if e.accepted and e.url})
        remember_finding(topic=job.query, summary=summary, source_urls=urls)

    return report
