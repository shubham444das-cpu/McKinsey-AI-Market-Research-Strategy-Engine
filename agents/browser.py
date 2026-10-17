"""
Browser / source-collection agent.

For each planner task, runs a Google search (see utils/search.py) and fetches
the top results' page text. Search failures are retried with a simplified
query and then with the plain sub-question; page fetches run in parallel and
fall back to the search snippet when a site blocks us.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

from utils.models import ResearchJob, SourceRecord
from utils.search import google_search

_MAX_RESULTS_PER_TASK = 4
_FETCH_TIMEOUT = (4, 6)    # (connect, read) seconds per page
_FETCH_DEADLINE = 25       # max seconds to wait for ALL pages together
_MAX_CHARS = 8000          # page text kept for the extractor
_INTER_QUERY_DELAY = 0.5   # pause before the retry of an empty search
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_BARE_DOMAIN = re.compile(r"^[\w.-]+\.[a-z]{2,}$", re.I)


def _fetch_page(url: str) -> Tuple[str, str]:
    """Return (page_title, visible_text). Raises on HTTP/non-HTML problems."""
    resp = requests.get(url, headers=_HEADERS, timeout=_FETCH_TIMEOUT)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" not in ctype and "text" not in ctype:
        raise ValueError(f"unsupported content type: {ctype or 'unknown'}")

    soup = BeautifulSoup(resp.content, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return title, text[:_MAX_CHARS]


def _search_with_fallbacks(job: ResearchJob, query: str, question: str) -> List[dict]:
    clean = " ".join(query.replace('"', "").replace("'", "").split())
    attempts = [clean]
    if question and question.strip() and question.strip() != clean:
        attempts.append(question.strip())
    errors: List[str] = []
    for i, q in enumerate(attempts):
        try:
            results = google_search(q, max_results=_MAX_RESULTS_PER_TASK)
            if results:
                if i > 0:
                    job.add_log(f"Search for '{clean}' only worked after falling back to the plain question.")
                return results
        except Exception as e:  # noqa: BLE001
            errors.append(str(e)[:300])
        if i < len(attempts) - 1:
            time.sleep(_INTER_QUERY_DELAY)
    if errors:
        job.add_log(f"Search failed for '{clean}': {errors[-1]}")
        job.search_errors.extend(errors)
    else:
        job.add_log(f"Search returned no results for '{clean}'.")
    return []


def collect_sources(job: ResearchJob) -> List[SourceRecord]:
    if job.plan is None:
        raise RuntimeError("Cannot browse before a plan exists.")

    pending: List[SourceRecord] = []
    seen_urls = set()

    tasks = [t for t in job.plan.tasks if (t.search_query or t.question).strip()]

    # Run all sub-question searches at the same time instead of one after another.
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(tasks)))) as pool:
        all_results = list(
            pool.map(lambda t: _search_with_fallbacks(job, t.search_query or t.question, t.question), tasks)
        )

    for task, results in zip(tasks, all_results):
        for r in results:
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            pending.append(
                SourceRecord(
                    task_id=task.id,
                    url=url,
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                )
            )

    def _fill(record: SourceRecord) -> SourceRecord:
        try:
            title, text = _fetch_page(record.url)
            if title and (not record.title or _BARE_DOMAIN.match(record.title)):
                record.title = title
            if len(text) < 200 and record.snippet:  # JS-only/empty page: keep snippet too
                text = f"{record.snippet}\n{text}".strip()
            record.raw_text = text
            record.fetch_ok = True
        except Exception as e:  # noqa: BLE001
            record.fetch_ok = False
            record.fetch_error = str(e)[:200]
            record.raw_text = record.snippet
        return record

    # Fetch every page in parallel, but never wait longer than _FETCH_DEADLINE overall;
    # pages that are still loading then fall back to their search snippet.
    pool = ThreadPoolExecutor(max_workers=12)
    futures = {pool.submit(_fill, rec): rec for rec in pending}
    wait(futures, timeout=_FETCH_DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)
    collected = []
    for fut, rec in futures.items():
        if fut.cancelled() or not fut.done():
            rec.fetch_ok = False
            rec.fetch_error = "timed out" if not fut.cancelled() else "skipped (fetch deadline reached)"
            rec.raw_text = rec.snippet
        collected.append(rec)

    job.sources = collected
    failed = sum(1 for s in collected if not s.fetch_ok)
    job.add_log(
        f"Browser agents collected {len(collected)} sources "
        f"({failed} failed to fetch full page; snippet used instead)."
    )
    if not collected:
        uniq = list(dict.fromkeys(job.search_errors))[:3]
        why = " | ".join(uniq) if uniq else "every provider returned zero results"
        raise RuntimeError(
            f"No sources found. Provider errors: {why}. "
            "Run `python check_search.py` to test each search provider on its own."
        )
    return collected
