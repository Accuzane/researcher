"""Retrieval: search the web and come back with readable documents.

The important change from a plain search call is `include_raw_content`. Search
snippets are ~200 characters of context-free text; asking a model to synthesise
research from three of them is the single biggest quality limit in a naive
pipeline. Here every result carries the extracted page body, truncated to a
budget, and lands in the registry as a `Source` with a stable id.

The backend is swappable via `set_backend` so the graph can be exercised in
tests without network access or an API key.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Protocol

from config import settings
from state import Source, source_id

logger = logging.getLogger("search")

RawResult = dict


class SearchBackend(Protocol):
    def __call__(self, query: str, max_results: int) -> list[RawResult]: ...


# --------------------------------------------------------------------------
# tavily backend
# --------------------------------------------------------------------------

_tavily = None


def _tavily_backend(query: str, max_results: int) -> list[RawResult]:
    global _tavily
    if _tavily is None:
        if not os.getenv("TAVILY_API_KEY"):
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Put it in .env, or call "
                "tools.search.set_backend() with your own retrieval function."
            )
        from langchain_tavily import TavilySearch

        _tavily = TavilySearch(
            max_results=max_results,
            include_raw_content=True,
            search_depth="advanced",
        )

    response = _tavily.invoke({"query": query})

    # The wrapper has returned both a dict envelope and a bare list across
    # versions; accept either rather than pinning to one shape.
    if isinstance(response, dict):
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        results = []
    return [r for r in results if isinstance(r, dict)]


_backend: SearchBackend = _tavily_backend


def set_backend(backend: SearchBackend) -> None:
    global _backend
    _backend = backend


def reset_backend() -> None:
    global _backend
    _backend = _tavily_backend


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def _best_text(result: RawResult) -> str:
    """Prefer the extracted page body; fall back to the search snippet."""
    for key in ("raw_content", "content", "text", "snippet"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def search(query: str, max_results: int | None = None) -> list[Source]:
    """Run one search and return it as `Source` records.

    Never raises on a backend failure -- a research run should lose one query,
    not collapse. Failures are logged and come back as an empty list.
    """
    limit = max_results or settings.results_per_query
    try:
        raw = _backend(query=query, max_results=limit)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("search failed for %r: %s", query, exc)
        return []

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources: list[Source] = []
    seen: set[str] = set()

    for result in raw[:limit]:
        url = (result.get("url") or "").strip()
        if not url:
            continue
        sid = source_id(url)
        if sid in seen:
            continue
        text = _best_text(result)
        if not text:
            continue
        seen.add(sid)
        sources.append(
            Source(
                id=sid,
                url=url,
                title=(result.get("title") or url).strip(),
                text=text[: settings.source_chars],
                query=query,
                retrieved_at=now,
            )
        )

    logger.info("search %r -> %d sources", query, len(sources))
    return sources


def search_many(queries: list[str], max_results: int | None = None) -> list[Source]:
    """Several queries, deduplicated by source id, order preserved."""
    out: list[Source] = []
    seen: set[str] = set()
    for query in queries:
        for src in search(query, max_results):
            if src["id"] not in seen:
                seen.add(src["id"])
                out.append(src)
    return out
