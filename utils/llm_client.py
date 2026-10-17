"""
Thin, single-purpose wrapper around the Gemini API.

Every agent that needs an LLM call (planner, extractor, validator,
report_writer, cross_verifier) goes through `ask_json` / `ask_text` here,
and anything that needs vector embeddings (utils.rag_store) goes through
`embed_texts`. Centralizing this makes it trivial to swap models/providers
or add cost/token logging in one place later — none of the agent code
needs to change.

Reliability: every call is wrapped in `_call_with_retry_and_fallback`,
which retries transient failures with exponential backoff and, if a model
keeps failing, falls back to the next model in GEMINI_MODEL_FALLBACK. This
matters because a multi-agent pipeline has many sequential LLM calls per
job (planner -> N extractions -> validation batches -> cross-verification
-> aggregation -> report) — without retry/fallback, one transient 503 deep
into a Deep-depth job wastes every call that came before it.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from the project root no matter where `streamlit run` is launched from.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)
load_dotenv()
logger = logging.getLogger(__name__)

# Models come from .env (GEMINI_MODEL / GEMINI_MODEL_FALLBACK / GEMINI_EMBED_MODEL).
# The defaults below are the same models as in the project's .env.example.
def _csv(value: str) -> List[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
_FALLBACK_MODELS = [
    m for m in _csv(os.getenv("GEMINI_MODEL_FALLBACK", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite"))
    if m != _MODEL
]
_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2").strip()

# Models that answered "not found / not supported" are skipped for the rest of the
# session instead of being retried on every single call (big time saver).
_dead_models: set = set()
_DEAD_MARKERS = ("404", "not found", "is not supported", "no longer available", "permission_denied", "403")

_MAX_ATTEMPTS_PER_MODEL = 2
# Gemini "thinking" models count reasoning tokens against max_output_tokens, so a
# small cap can cut the JSON answer off mid-way. Always leave generous headroom.
_MIN_OUTPUT_TOKENS = 8192
_BASE_BACKOFF_SECONDS = 1.0

_RETRYABLE_STATUS_CODES = {"429", "500", "502", "503", "504"}
_RETRYABLE_MARKERS = ("timeout", "temporarily", "unavailable", "deadline", "connection", "overloaded")

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (or empty) in .env. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90_000))
    return _client


def _has_status(msg: str, codes) -> bool:
    """True if any HTTP status code appears as a standalone number in msg."""
    return any(re.search(rf"(?<!\d){c}(?!\d)", msg) for c in codes)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if _has_status(msg, _RETRYABLE_STATUS_CODES):
        return True
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _call_with_retry_and_fallback(fn, models: Optional[List[str]] = None, mark_dead: bool = True):
    """
    Try fn(model) across `models` (default: [primary, *fallbacks]),
    retrying each model with exponential backoff before moving on to the
    next one. Non-retryable errors (bad request, auth, etc.) skip straight
    to the next model rather than burning retries on the same failure.
    Raises the last error only after every model/attempt is exhausted.

    mark_dead=False: don't add a model to the session-wide dead list on 403/404-style
    errors. Used by the search tool, where "grounding not allowed for this key/model"
    must not stop the planner/extractor from using that same model.
    """
    models = models or [_MODEL, *_FALLBACK_MODELS]
    last_exc: Optional[Exception] = None

    live = [m for m in models if m not in _dead_models]
    models = live or models  # if everything was marked dead, try them all again

    for model_index, model in enumerate(models):
        for attempt in range(_MAX_ATTEMPTS_PER_MODEL):
            try:
                result = fn(model)
                if model_index > 0 or attempt > 0:
                    logger.info("Gemini call succeeded on model=%s after %d attempt(s).", model, attempt + 1)
                return result
            except Exception as e:  # noqa: BLE001 - SDK/API error shapes vary; classify by message
                last_exc = e
                retryable = _is_retryable(e)
                logger.warning(
                    "Gemini call failed model=%s attempt=%d/%d retryable=%s: %s",
                    model, attempt + 1, _MAX_ATTEMPTS_PER_MODEL, retryable, e,
                )
                if not retryable:
                    emsg = str(e).lower()
                    if mark_dead and (
                        _has_status(emsg, ("403", "404"))
                        or any(mk in emsg for mk in _DEAD_MARKERS if not mk.isdigit())
                    ):
                        _dead_models.add(model)
                        logger.warning("Model %s unavailable for this key; skipping it from now on.", model)
                    break  # don't waste retries on a non-transient error; try next model
                if attempt < _MAX_ATTEMPTS_PER_MODEL - 1:
                    sleep_s = _BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
        if model_index < len(models) - 1:
            logger.warning("Falling back from model=%s to model=%s.", model, models[model_index + 1])

    raise RuntimeError(
        f"All Gemini models ({', '.join(models)}) failed after retries. Last error: {last_exc}"
    ) from last_exc


def ask_text(system: str, user: str, max_tokens: int = 1200) -> str:
    """Single-turn completion, returns plain text. Retries/falls back on failure."""
    client = _get_client()

    def _call(model: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max(max_tokens, _MIN_OUTPUT_TOKENS),
            ),
        )
        return (resp.text or "").strip()

    return _call_with_retry_and_fallback(_call)


def ask_json(system: str, user: str, max_tokens: int = 1500) -> Any:
    """
    Single-turn completion that expects a JSON object/array back.
    Uses Gemini's JSON response mode, with defensive fence-stripping.
    A malformed-JSON response is treated as retryable (a different model
    or another attempt at the same model may well succeed) rather than
    failing the whole pipeline stage outright.
    """
    client = _get_client()

    def _call(model: str) -> Any:
        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max(max_tokens, _MIN_OUTPUT_TOKENS),
                response_mime_type="application/json",
            ),
        )
        raw = (resp.text or "").strip()
        if not raw:
            raise ValueError(f"Model {model} returned an empty response (temporarily unavailable / blocked).")
        cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model {model} did not return valid JSON: {e}\nRaw output:\n{raw}") from e

    return _call_with_retry_and_fallback(_call)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch-embed a list of strings, returning one vector per input string
    (same order). Backs the RAG chunk/retrieval layer in utils.rag_store.
    Uses its own retry/fallback pass over embedding-model candidates only
    (chat models aren't valid fallbacks for an embedding call).
    """
    if not texts:
        return []
    client = _get_client()

    def _call(model: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), 90):  # API limit is 100 inputs per request
            resp = client.models.embed_content(model=model, contents=texts[i : i + 90])
            vectors.extend(list(e.values) for e in resp.embeddings)
        return vectors

    return _call_with_retry_and_fallback(_call, models=[_EMBED_MODEL])
