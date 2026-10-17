"""
Google web search for the browsing agent.

Google no longer offers a usable public search API (the Custom Search JSON
API is closed to new customers and shuts down on 2027-01-01), so this module
uses two Google-backed routes instead, tried in order:

  1. "gemini"       - Gemini's built-in *Grounding with Google Search* tool.
                      Uses your existing GEMINI_API_KEY, no extra setup, and is
                      not blocked the way scraping is.
  2. "googlesearch" - the `googlesearch-python` package, which reads Google's
                      public results page. No key, but Google may rate-limit it.

  3. "duckduckgo"  - keyless HTML scrape of DuckDuckGo. Not Google, but it is the
                      last-resort safety net: Google has started serving JS-only
                      pages to scrapers, which makes (2) return nothing.

Choose with SEARCH_PROVIDER in .env: auto (default) | gemini | googlesearch | duckduckgo.
`auto` tries them in the order above and stops at the first one that returns results.
Every function returns a list of dicts: {"url", "title", "snippet"}.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from utils.llm_client import _FALLBACK_MODELS, _MODEL, _call_with_retry_and_fallback, _get_client

logger = logging.getLogger(__name__)

SearchResult = Dict[str, str]

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ------------------------------------------------------------ Provider 1 --
def _resolve_redirect(url: str) -> str:
    """Grounding links are vertexaisearch redirect URLs; follow to the real page."""
    if "grounding-api-redirect" not in url:
        return url
    try:
        with requests.get(url, headers={"User-Agent": _UA}, allow_redirects=True,
                          stream=True, timeout=5) as r:
            return r.url or url
    except Exception:  # noqa: BLE001
        return url


# Grounded calls are slow and quota-limited; running 5 at once is what triggers 429s.
_GROUNDING_GATE = threading.Semaphore(2)
# Fail over to the next *provider* quickly instead of cycling through every model.
_GROUNDING_MODELS = [_MODEL, *_FALLBACK_MODELS][:2]


def _grounded_call(client, types, model: str, prompt: str):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=8192,  # thinking models spend part of this on reasoning
        ),
    )


def _extract_chunks(resp):
    if not getattr(resp, "candidates", None):
        return None, [], []
    meta = getattr(resp.candidates[0], "grounding_metadata", None)
    chunks = (getattr(meta, "grounding_chunks", None) or []) if meta else []
    supports = (getattr(meta, "grounding_supports", None) or []) if meta else []
    return meta, chunks, supports


def _search_gemini_grounding(query: str, max_results: int) -> List[SearchResult]:
    from google.genai import types

    client = _get_client()
    prompts = [
        # 1st try: plain request
        f"Search the web for: {query}\n"
        "Summarise the most relevant, recent and authoritative findings, "
        "citing specific facts and figures.",
        # 2nd try (only if the model answered without searching): be explicit
        f"You MUST use Google Search to answer. Query: {query}\n"
        "Do not answer from memory. Find current web pages and report their key facts.",
    ]

    resp, chunks, supports = None, [], []
    for prompt in prompts:
        with _GROUNDING_GATE:
            resp = _call_with_retry_and_fallback(
                lambda m, _p=prompt: _grounded_call(client, types, m, _p),
                models=_GROUNDING_MODELS,
                mark_dead=False,
            )
        _meta, chunks, supports = _extract_chunks(resp)
        if chunks:
            break
    if not chunks:
        return []

    # Build a per-source snippet from the answer sentences each source supports.
    snippets: Dict[int, List[str]] = {}
    for sup in supports:
        text = getattr(getattr(sup, "segment", None), "text", "") or ""
        for idx in getattr(sup, "grounding_chunk_indices", None) or []:
            snippets.setdefault(idx, []).append(text)
    try:
        fallback_snippet = (resp.text or "")[:600]
    except Exception:  # noqa: BLE001 - .text can raise if the reply has no text part
        fallback_snippet = ""

    candidates = []
    for i, ch in enumerate(chunks):
        web = getattr(ch, "web", None)
        uri = getattr(web, "uri", None) if web else None
        if uri:
            candidates.append((i, web, uri))
    candidates = candidates[: max_results * 2]  # a few spares in case of duplicates
    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        resolved = list(pool.map(lambda c: _resolve_redirect(c[2]), candidates))

    results: List[SearchResult] = []
    seen = set()
    for (i, web, _), url in zip(candidates, resolved):
        if url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": (getattr(web, "title", "") or urlparse(url).netloc),
                "snippet": " ".join(snippets.get(i, []))[:800] or fallback_snippet,
            }
        )
        if len(results) >= max_results:
            break
    return results


# ------------------------------------------------------------ Provider 2 --
def _search_googlesearch(query: str, max_results: int) -> List[SearchResult]:
    from googlesearch import search  # pip install googlesearch-python

    results: List[SearchResult] = []
    for r in search(query, num_results=max_results, advanced=True, sleep_interval=0, timeout=6):
        url = getattr(r, "url", "") or ""
        if not url.startswith("http"):
            continue
        results.append(
            {
                "url": url,
                "title": getattr(r, "title", "") or urlparse(url).netloc,
                "snippet": getattr(r, "description", "") or "",
            }
        )
    return results


# ------------------------------------------------------------ Provider 3 --
_DDG_ENDPOINTS = ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/")


def _ddg_real_url(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded real url>."""
    if href.startswith("//"):
        href = "https:" + href
    q = parse_qs(urlparse(href).query)
    if "uddg" in q and q["uddg"]:
        return unquote(q["uddg"][0])
    return href


def _parse_ddg_html(html: str, max_results: int) -> List[SearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[SearchResult] = []
    seen = set()

    # html.duckduckgo.com layout
    for a in soup.select("a.result__a"):
        url = _ddg_real_url(a.get("href", ""))
        if not url.startswith("http") or "duckduckgo.com" in urlparse(url).netloc or url in seen:
            continue
        body = a.find_parent(class_="result")
        sn = body.select_one(".result__snippet") if body else None
        seen.add(url)
        out.append({"url": url, "title": a.get_text(" ", strip=True) or urlparse(url).netloc,
                    "snippet": sn.get_text(" ", strip=True) if sn else ""})
        if len(out) >= max_results:
            return out

    # lite.duckduckgo.com layout (table rows: link row, then snippet row)
    for a in soup.select("a.result-link"):
        url = _ddg_real_url(a.get("href", ""))
        if not url.startswith("http") or "duckduckgo.com" in urlparse(url).netloc or url in seen:
            continue
        row = a.find_parent("tr")
        nxt = row.find_next_sibling("tr") if row else None
        sn = nxt.select_one(".result-snippet") if nxt else None
        seen.add(url)
        out.append({"url": url, "title": a.get_text(" ", strip=True) or urlparse(url).netloc,
                    "snippet": sn.get_text(" ", strip=True) if sn else ""})
        if len(out) >= max_results:
            break
    return out


def _search_duckduckgo(query: str, max_results: int) -> List[SearchResult]:
    last = "no response"
    for endpoint in _DDG_ENDPOINTS:
        try:
            r = requests.post(endpoint, data={"q": query}, headers={"User-Agent": _UA}, timeout=10)
            if r.status_code != 200:
                last = f"HTTP {r.status_code} from {urlparse(endpoint).netloc}"
                continue
            results = _parse_ddg_html(r.text, max_results)
            if results:
                return results
            last = f"0 results parsed from {urlparse(endpoint).netloc} (blocked or layout changed)"
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(last)


_PROVIDERS = {
    "gemini": _search_gemini_grounding,
    "googlesearch": _search_googlesearch,
    "duckduckgo": _search_duckduckgo,
}
_AUTO_ORDER = ["gemini", "googlesearch", "duckduckgo"]


def google_search(query: str, max_results: int = 4) -> List[SearchResult]:
    """Search Google. Raises RuntimeError (with per-provider reasons) if all fail."""
    choice = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()
    order = _AUTO_ORDER if choice == "auto" else [choice]
    errors: List[str] = []
    for name in order:
        fn = _PROVIDERS.get(name)
        if fn is None:
            errors.append(f"unknown SEARCH_PROVIDER '{name}'")
            continue
        try:
            results = fn(query, max_results)
            if results:
                return results
            errors.append(f"{name}: no results")
        except ImportError as e:
            errors.append(f"{name}: missing package ({e}) - run pip install -r requirements.txt")
        except Exception as e:  # noqa: BLE001
            logger.warning("Search provider %s failed: %s", name, e)
            errors.append(f"{name}: {str(e)[:200]}")
    raise RuntimeError("; ".join(errors))
