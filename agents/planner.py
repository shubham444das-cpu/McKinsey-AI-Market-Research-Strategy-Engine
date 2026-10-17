"""
Planner agent.

Takes the raw research intake and produces a ResearchPlan: a small set of
focused sub-questions, each with its own search query and expected
evidence type. This is what lets the browsing stage run several targeted
searches instead of one vague one.
"""
from __future__ import annotations

from utils.llm_client import ask_json
from utils.models import ResearchJob, ResearchPlan, ResearchTask

_DEPTH_TASK_COUNT = {"Quick": 3, "Standard": 5, "Deep": 8}


SYSTEM_PROMPT = """You are a senior strategy research planner at a consulting firm.
Given a client research request, break it into a small number of focused,
independently-searchable sub-questions. Each sub-question should map to ONE
good web search query. Cover market signals, competitor moves, and trends
relevant to the request — don't pad with filler questions.
"""


def build_plan(job: ResearchJob) -> ResearchPlan:
    task_count = _DEPTH_TASK_COUNT.get(job.depth, 5)

    user_prompt = f"""
Research request: {job.query}
Market/sector: {job.market or "not specified"}
Geography: {job.geography or "not specified"}
Timeframe of interest: {job.timeframe or "not specified"}
Named competitors: {job.competitors or "none specified"}

Produce exactly {task_count} sub-questions.

Return a JSON object of this shape:
{{
  "objective": "<one sentence restating the research objective>",
  "tasks": [
    {{
      "question": "<the sub-question>",
      "search_query": "<a concrete web search query string>",
      "expected_evidence_type": "<short label, e.g. 'market size stat', 'competitor product launch'>"
    }}
  ]
}}
"""
    data = ask_json(SYSTEM_PROMPT, user_prompt)
    if isinstance(data, list):  # model returned the tasks array directly
        data = {"tasks": data}
    if not isinstance(data, dict):
        data = {}

    plan = ResearchPlan(objective=data.get("objective") or job.query)
    for t in data.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        question = t.get("question", "") or ""
        query = t.get("search_query", "") or question
        if not query.strip():
            continue
        plan.tasks.append(
            ResearchTask(
                question=question,
                search_query=query,
                expected_evidence_type=t.get("expected_evidence_type", ""),
            )
        )
    if not plan.tasks:
        raise RuntimeError("Planner returned no usable research tasks - please try again.")

    job.plan = plan
    job.add_log(f"Planner produced {len(plan.tasks)} research tasks.")
    return plan
