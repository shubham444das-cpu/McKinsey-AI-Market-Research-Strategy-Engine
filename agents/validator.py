"""
Validation agent.

Checks each evidence record for: internal plausibility, whether it's a
near-duplicate of another record, and whether the claim goes meaningfully
beyond what the excerpt actually supports (unsupported-claim guard). Flags
rather than silently drops low-confidence evidence, per the dossier's
requirement that low-confidence evidence be visible, not hidden.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import List

from utils.llm_client import ask_json
from utils.models import ResearchJob, EvidenceRecord

SYSTEM_PROMPT = """You are a fact-validation agent for a strategy research
pipeline. For each claim/excerpt pair, judge whether the excerpt actually
supports the claim, and how confident a careful analyst should be that the
claim is accurately represented. Be skeptical of claims that go beyond what
the excerpt states.
"""

_DUPLICATE_THRESHOLD = 0.85


def _mark_duplicates(evidence: List[EvidenceRecord]) -> None:
    for i, a in enumerate(evidence):
        if not a.accepted:
            continue
        for b in evidence[i + 1 :]:
            if not b.accepted:
                continue
            similarity = SequenceMatcher(None, a.claim.lower(), b.claim.lower()).ratio()
            if similarity >= _DUPLICATE_THRESHOLD:
                b.flags.append("duplicate")
                b.validation_notes = (b.validation_notes + " Near-duplicate of another claim.").strip()
                b.confidence = min(b.confidence, 0.3)


def _validate_batch(evidence: List[EvidenceRecord]) -> None:
    if not evidence:
        return

    items_payload = [
        {"id": e.id, "claim": e.claim, "excerpt": e.excerpt} for e in evidence
    ]
    user_prompt = f"""
Evidence items to validate:
{items_payload}

For each item, return a JSON array of:
{{
  "id": "<matching id>",
  "confidence": <float 0.0-1.0>,
  "flags": [<zero or more of: "unsupported", "vague", "opinion_not_fact", "stale">],
  "notes": "<one short sentence explaining the score>"
}}
"""
    try:
        results = ask_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)
    except Exception:
        # fail safe: leave default low-ish confidence rather than crash the pipeline
        for e in evidence:
            e.confidence = 0.4
            e.validation_notes = "Validator call failed; default confidence applied."
        return

    by_id = {e.id: e for e in evidence}
    seen = set()
    for r in results if isinstance(results, list) else []:
        if not isinstance(r, dict):
            continue
        e = by_id.get(r.get("id"))
        if not e:
            continue
        try:
            e.confidence = max(0.0, min(1.0, float(r.get("confidence", 0.4))))
        except (TypeError, ValueError):
            e.confidence = 0.4
        flags = r.get("flags") or []
        if isinstance(flags, list):
            e.flags.extend(f for f in flags if isinstance(f, str))
        e.validation_notes = str(r.get("notes", ""))
        seen.add(e.id)
    for e in evidence:  # items the model skipped: neutral-low default instead of 0.0
        if e.id not in seen:
            e.confidence = 0.4
            e.validation_notes = "Validator returned no score for this item; default applied."


def validate_evidence(job: ResearchJob) -> List[EvidenceRecord]:
    evidence = job.evidence

    # batch in chunks so one call doesn't get too large for deep research jobs
    chunk_size = 12
    batches = [evidence[i : i + chunk_size] for i in range(0, len(evidence), chunk_size)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_validate_batch, batches))

    _mark_duplicates(evidence)

    for e in evidence:
        e.accepted = e.confidence >= 0.5 and "duplicate" not in e.flags

    accepted = sum(1 for e in evidence if e.accepted)
    job.add_log(
        f"Validation complete: {accepted}/{len(evidence)} evidence records accepted "
        f"(confidence >= 0.5, not duplicates)."
    )
    return evidence
