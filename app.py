"""
AI Market Research & Strategy Engine — Streamlit edition.

One process, one UI, no separate backend service: Streamlit calls the
agent functions directly and stores pipeline state in st.session_state.
"""
from __future__ import annotations

import streamlit as st

from utils.models import ResearchJob
from utils import memory
from agents import planner, browser, extractor, validator, aggregator, report_writer

st.set_page_config(page_title="AI Market Research & Strategy Engine", layout="wide")

if "job" not in st.session_state:
    st.session_state.job = None


def _job() -> ResearchJob | None:
    return st.session_state.job


def _run_pipeline(job: ResearchJob, progress_placeholder) -> None:
    steps = [
        ("planning", "Planning research tasks...", planner.build_plan),
        ("browsing", "Searching and fetching sources...", browser.collect_sources),
        ("extraction", "Extracting structured evidence...", extractor.extract_evidence),
        ("validation", "Validating evidence quality...", validator.validate_evidence),
        ("aggregation", "Grouping evidence into themes...", aggregator.aggregate),
        ("report", "Writing the strategy brief...", report_writer.write_report),
    ]
    with progress_placeholder.container():
        for status, label, fn in steps:
            job.status = status
            with st.status(label, expanded=False) as box:
                try:
                    fn(job)
                except Exception:
                    box.update(label=f"Failed: {label}", state="error")
                    raise
                box.update(label=label.replace("...", " - done"), state="complete")
    job.status = "done"


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("Knowledge & Memory")
    records = memory.all_records()
    st.caption(f"{len(records)} validated finding(s) stored from prior research jobs.")
    if records:
        for r in records[-5:][::-1]:
            with st.expander(r["topic"][:60] or "(untitled)"):
                st.write(r["summary"])
                for u in r["source_urls"][:5]:
                    st.markdown(f"- [{u}]({u})")
    st.divider()
    st.caption(
        "This is a first-draft research accelerator, not client-ready advice. "
        "Every generated brief requires human review before use."
    )

st.title("AI Market Research & Strategy Engine")
st.caption("Query → Planner → Web Search Agents → Extraction → Validation → Aggregation → Report")

tab_intake, tab_monitor, tab_evidence, tab_report = st.tabs(
    ["1. Research Intake", "2. Workflow Monitor", "3. Evidence Review", "4. Report"]
)

# ----------------------------------------------------------- Intake tab --
with tab_intake:
    st.subheader("Research Intake Console")
    with st.form("intake_form"):
        query = st.text_area(
            "Research question",
            placeholder="e.g. Should we enter the fintech-for-freelancers market in Southeast Asia?",
            height=100,
        )
        col1, col2 = st.columns(2)
        with col1:
            market = st.text_input("Market / sector")
            geography = st.text_input("Geography")
        with col2:
            timeframe = st.text_input("Timeframe of interest (e.g. last 12 months)")
            competitors = st.text_input("Named competitors (comma-separated)")
        depth = st.select_slider("Research depth", options=["Quick", "Standard", "Deep"], value="Standard")

        submitted = st.form_submit_button("Run research pipeline", type="primary")

    if submitted:
        if not query.strip():
            st.error("Please enter a research question before running the pipeline.")
        else:
            job = ResearchJob(
                query=query.strip(),
                market=market.strip(),
                geography=geography.strip(),
                timeframe=timeframe.strip(),
                competitors=competitors.strip(),
                depth=depth,
            )
            st.session_state.job = job
            progress_placeholder = st.empty()
            try:
                _run_pipeline(job, progress_placeholder)
                if not any(e.accepted for e in job.evidence):
                    st.warning("Pipeline finished, but no evidence passed validation - the report will be thin. "
                               "Check the Workflow Monitor log.")
                else:
                    st.success("Pipeline complete — see the Workflow Monitor, Evidence Review, and Report tabs.")
            except Exception as e:
                job.status = "failed"
                job.add_log(f"Pipeline error: {e}")
                st.error(f"Pipeline failed: {e}")

    with st.expander("Sample research requests"):
        st.markdown(
            "- *Market-entry scan*: \"Should we enter the fintech-for-freelancers "
            "market in Southeast Asia?\"\n"
            "- *Competitor landscape*: \"How are Stripe, Adyen, and Checkout.com "
            "positioning in embedded finance?\"\n"
            "- *Trend monitoring*: \"What are the emerging signals in enterprise "
            "AI agent adoption this year?\""
        )

# ---------------------------------------------------------- Monitor tab --
with tab_monitor:
    st.subheader("Agent Workflow Monitor")
    job = _job()
    if not job:
        st.info("Run a research job from the Intake tab to see workflow progress here.")
    else:
        st.write(f"**Job ID:** `{job.id}`  |  **Status:** `{job.status}`")

        if job.plan:
            st.markdown("**Research plan**")
            for t in job.plan.tasks:
                st.markdown(f"- *{t.question}* → search: `{t.search_query}` ({t.expected_evidence_type})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sources collected", len(job.sources))
        m2.metric("Evidence extracted", len(job.evidence))
        m3.metric("Evidence accepted", sum(1 for e in job.evidence if e.accepted))
        m4.metric("Themes", len(job.themes))

        if job.related_memory:
            st.markdown("**Related prior research (RAG memory recall)**")
            for r in job.related_memory:
                with st.expander(r["topic"][:60] or "(untitled)"):
                    st.write(r["summary"])
                    for u in r["source_urls"][:5]:
                        st.markdown(f"- [{u}]({u})")

        st.markdown("**Execution log**")
        st.code("\n".join(job.log) or "No log entries yet.", language=None)

# --------------------------------------------------------- Evidence tab --
with tab_evidence:
    st.subheader("Evidence Review Panel")
    job = _job()
    if not job or not job.evidence:
        st.info("No evidence yet — run a research job first.")
    else:
        show_flagged_only = st.checkbox("Show only flagged / rejected evidence")
        rows = job.evidence
        if show_flagged_only:
            rows = [e for e in rows if not e.accepted or e.flags]

        for e in rows:
            badge = "✅ accepted" if e.accepted else "⛔ excluded"
            with st.expander(f"{badge} · conf {e.confidence:.2f} · {e.claim[:90]}"):
                st.write(f"**Claim:** {e.claim}")
                st.write(f"**Excerpt:** {e.excerpt}")
                st.write(f"**Entity:** {e.entity}  |  **Topic:** {e.topic}")
                st.write(f"**Source:** [{e.url}]({e.url})")
                if e.flags:
                    st.warning(f"Flags: {', '.join(e.flags)}")
                if e.validation_notes:
                    st.caption(e.validation_notes)

        st.divider()
        st.markdown("**Source fetch status**")
        failed_sources = [s for s in job.sources if not s.fetch_ok]
        st.write(f"{len(job.sources) - len(failed_sources)} of {len(job.sources)} pages fetched successfully.")
        if failed_sources:
            with st.expander(f"{len(failed_sources)} source(s) failed to fetch (fell back to search snippet)"):
                for s in failed_sources:
                    st.write(f"- [{s.title or s.url}]({s.url}) — {s.fetch_error}")

# ----------------------------------------------------------- Report tab --
with tab_report:
    st.subheader("Strategy Brief")
    job = _job()
    if not job or not job.report:
        st.info("No report yet — run a research job first.")
    else:
        r = job.report
        st.markdown(f"### Executive Summary\n{r.executive_summary}")
        st.markdown(f"### Key Findings\n{r.key_findings}")
        st.markdown(f"### Market Signals\n{r.market_signals}")
        st.markdown(f"### Competitor Observations\n{r.competitor_observations}")
        st.markdown(f"### Implications\n{r.implications}")
        st.markdown(f"### Recommendations\n{r.recommendations}")
        st.info(r.human_review_note)

        with st.expander("Evidence appendix (all extracted evidence, including excluded)"):
            st.markdown(r.evidence_appendix_md)

        full_md = (
            f"# Strategy Research Brief\n\n**Query:** {job.query}\n\n"
            f"## Executive Summary\n{r.executive_summary}\n\n"
            f"## Key Findings\n{r.key_findings}\n\n"
            f"## Market Signals\n{r.market_signals}\n\n"
            f"## Competitor Observations\n{r.competitor_observations}\n\n"
            f"## Implications\n{r.implications}\n\n"
            f"## Recommendations\n{r.recommendations}\n\n"
            f"## Evidence Appendix\n{r.evidence_appendix_md}\n\n"
            f"---\n{r.human_review_note}\n"
        )
        st.download_button(
            "Download report as Markdown",
            data=full_md,
            file_name=f"research_brief_{job.id}.md",
            mime="text/markdown",
        )
