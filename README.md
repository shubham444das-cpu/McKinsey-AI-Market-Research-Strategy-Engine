# AI Market Research & Strategy Engine (Streamlit Edition)

A single-stack, Python-only implementation of an agentic market-research pipeline:

```
Query -> Planner Agent -> Web Search Agents -> Extraction -> Validation -> Aggregation -> Report Generation
```

Everything — UI and "backend" logic — runs inside one Streamlit app and a set of plain
Python modules. There is no separate JS/Node backend and no Java anywhere in the stack.

## Why this shape

The original dossier calls for Next.js + FastAPI + multi-agent orchestration. For a
data-science/GenAI project you don't need three languages to prove the workflow works.
Streamlit gives you the UI, session state gives you the "job" object, and Python
functions give you the agents. You can always peel this apart into a real frontend/backend
later if the product needs to scale to many concurrent users — the agent modules are
already isolated so that split is mechanical, not a rewrite.

## Architecture

```
app.py                     <- Streamlit UI (4 tabs = 4 product surfaces)
agents/
  planner.py                <- decomposes the research question into sub-tasks
  browser.py                 <- runs Google search + fetches pages for each sub-task
  extractor.py                <- turns page text into structured evidence records (LLM)
  validator.py                 <- scores each evidence record for credibility/recency/conflict
  aggregator.py                 <- groups validated evidence into themes
  report_writer.py               <- writes the final Markdown research brief (LLM)
utils/
  models.py                  <- dataclasses: ResearchJob, SourceRecord, EvidenceRecord, Report
  llm_client.py                <- Gemini API wrapper, with retry + model fallback (see below)
  rag_store.py                   <- chunk/embed/retrieve layer (traditional RAG, see below)
  memory.py                        <- local JSON "memory" of past validated findings, for reuse
data/
  sample_research_requests.json
```

## Reliability: Gemini retry and fallback

Every LLM call (`ask_text`, `ask_json`, `embed_texts` in `utils/llm_client.py`)
goes through `_call_with_retry_and_fallback`, which:

- retries transient failures (429/500/502/503/504, timeouts, "unavailable")
  on the same model with exponential backoff, up to 3 attempts;
- on non-retryable errors, or once retries on a model are exhausted, moves
  on to the next model in `GEMINI_MODEL_FALLBACK` (comma-separated,
  configurable in `.env`) and retries there;
- only raises once every model/attempt combination has failed.

A malformed-JSON response from `ask_json` is treated as retryable too —
another attempt or another model may well produce valid JSON — instead of
failing the whole pipeline stage on one bad parse.

## Retrieval-augmented generation (traditional RAG)

`utils/rag_store.py` implements a small, dependency-light RAG layer: chunk
text -> embed chunks (Gemini embeddings) -> brute-force cosine-similarity
retrieval -> feed the retrieved text into an LLM prompt. It's used in three
places:

- **Extraction** (`agents/extractor.py`): long source pages are chunked and
  only the chunks most relevant to the specific sub-question are retrieved
  and passed to the extraction prompt, instead of blindly truncating the
  page to the first N characters.
- **Aggregation** (`agents/aggregator.py`): if a Deep-depth job produces
  more accepted evidence than comfortably fits one synthesis prompt, the
  claims most relevant to the research objective are retrieved instead of
  dumping every claim in. (The report's evidence appendix still lists every
  accepted/excluded record regardless — this only affects what goes into
  the theme-synthesis prompt.)
- **Memory** (`utils/memory.py`): past validated findings are embedded and
  stored in `rag_store.json`; a new query recalls related past findings by
  embedding similarity rather than raw keyword overlap.

Swap `rag_store.py`'s persistence/index for FAISS/Chroma/pgvector later if
you need it to scale past one process's in-memory corpus — the chunk/embed/
retrieve call shapes (`chunk_text`, `rank_by_relevance`, `persist_finding`,
`recall_similar`) are the only things callers depend on.

Job state machine (mirrors the dossier's lifecycle, just in-process instead of a queue):

`intake -> planning -> browsing -> extraction -> validation -> aggregation -> report -> done`

Each stage's output is stored in `st.session_state.job` so the Workflow Monitor tab can
show exactly what happened at each step (sources found, evidence extracted, what was
flagged low-confidence, what was rejected) — that transparency is the whole point of the
Evidence Review Panel in the original dossier.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your key (get one at
   https://aistudio.google.com/apikey):
   ```
   GEMINI_API_KEY=...
   ```
3. `streamlit run app.py`

No other services required — search (`utils/search.py`) tries Gemini's Google Search grounding with your
existing key, then `googlesearch-python`, then a keyless DuckDuckGo scrape; set `SEARCH_PROVIDER` in `.env`,
and run `python check_search.py` to see which provider works on your machine)
and page fetching uses `requests` + `BeautifulSoup`. Memory is a local JSON file
(`memory_store.json`), so there's no database to stand up. Swap in Chroma/Qdrant/Supabase
later if you need multi-user persistence — `utils/memory.py` is the only file that
would need to change.

## Product surfaces (mapped to Streamlit tabs)

| Dossier surface | Streamlit tab |
|---|---|
| Research Intake Console | "1. Research Intake" |
| Agent Workflow Monitor | "2. Workflow Monitor" |
| Evidence Review Panel | "3. Evidence Review" |
| Strategy Brief Builder | "4. Report" |

The Knowledge/Memory Workspace and Ops Dashboard from the dossier are represented as a
sidebar memory browser and a small stats panel — full versions of those are natural
follow-ups once the core pipeline is proven out.

## Honest scope notes

- Source "validation" is LLM-judged plausibility + recency + duplication checks, not a
  hardened credibility database — flag this clearly to reviewers, same as the dossier
  asks.
- This is a first-draft generator. Every report ends with a human-review reminder and
  never claims to be client-ready advice, per the dossier's compliance requirement.
- No client-sensitive or credential data is stored anywhere; memory only holds
  research topic summaries and public-source citations.
