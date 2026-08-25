"""Shared state for the research graph.

The reducers here are load-bearing. Sub-researchers run concurrently via
`Send`, so every key they write needs a reducer that can merge two
independently-produced updates. Without one, LangGraph raises on concurrent
writes to the same key -- and worse, a last-write-wins key would silently drop
one branch's findings.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlsplit, urlunsplit


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

class Source(TypedDict):
    """One retrieved document. The unit of evidence in this system.

    `id` is derived from the normalised URL, so two sub-researchers that land
    on the same page independently produce the same id and the registry
    deduplicates them without any coordination between the branches.
    """
    id: str
    url: str
    title: str
    text: str
    query: str          # the search query that surfaced this
    retrieved_at: str


class SubQuestion(TypedDict):
    id: str
    question: str
    status: Literal["pending", "answered", "unanswered"]
    findings: str       # prose, carrying [S…] citation markers
    source_ids: list[str]
    round: int          # which research round produced this


class Gap(TypedDict):
    sub_question_id: str
    reason: str
    follow_up_query: str


def normalise_url(url: str) -> str:
    """Collapse trivial URL variants so the same page gets one id.

    Drops the fragment, lowercases scheme and host, strips a trailing slash.
    Deliberately conservative: query strings are preserved, because for plenty
    of sites they select the actual content.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        parts.query,
        "",
    ))


def source_id(url: str) -> str:
    """Stable, collision-resistant id for a URL. Deterministic across processes."""
    digest = hashlib.sha1(normalise_url(url).encode("utf-8")).hexdigest()
    return f"S{digest[:8]}"


# --------------------------------------------------------------------------
# reducers
# --------------------------------------------------------------------------

def merge_sources(left: list[Source] | None, right: list[Source] | None) -> list[Source]:
    """Union by source id, first write wins, insertion order preserved."""
    out: list[Source] = []
    seen: set[str] = set()
    for bucket in (left or [], right or []):
        for src in bucket:
            sid = src.get("id") or source_id(src.get("url", ""))
            if sid in seen:
                continue
            seen.add(sid)
            out.append({**src, "id": sid})
    return out


def merge_sub_questions(
    left: list[SubQuestion] | None,
    right: list[SubQuestion] | None,
) -> list[SubQuestion]:
    """Merge by sub-question id.

    A later update wins field by field, but only for fields it actually filled
    in -- so a sub-researcher returning just `findings` can't blank out the
    `question` text it was working from.
    """
    merged: dict[str, SubQuestion] = {}
    order: list[str] = []
    for bucket in (left or [], right or []):
        for sq in bucket:
            sid = sq["id"]
            if sid not in merged:
                merged[sid] = dict(sq)  # type: ignore[assignment]
                order.append(sid)
            else:
                for key, value in sq.items():
                    if value not in (None, "", []):
                        merged[sid][key] = value  # type: ignore[literal-required]
    return [merged[sid] for sid in order]


def add_int(left: int | None, right: int | None) -> int:
    return (left or 0) + (right or 0)


# --------------------------------------------------------------------------
# graph state
# --------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # input
    query: str

    # planning
    topic: str
    brief: str
    sub_questions: Annotated[list[SubQuestion], merge_sub_questions]

    # evidence
    sources: Annotated[list[Source], merge_sources]
    search_count: Annotated[int, add_int]

    # iteration control
    round: int
    gaps: list[Gap]
    stop_reason: str

    # output
    report: str
    citation_issues: list[str]
    saved_path: str

    # routing (top-level graph only)
    route: str
    answer: str


def new_state(query: str) -> ResearchState:
    """A clean starting state. Never reuse a dict between runs."""
    return {
        "query": query,
        "topic": "",
        "brief": "",
        "sub_questions": [],
        "sources": [],
        "search_count": 0,
        "round": 0,
        "gaps": [],
        "stop_reason": "",
        "report": "",
        "citation_issues": [],
        "saved_path": "",
        "route": "",
        "answer": "",
    }
