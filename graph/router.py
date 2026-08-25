"""Top-level routing: research the web, or answer from saved notes.

Deliberately a plain function rather than a graph edge. Routing on a raw model
string is how you get a run that dies on `"researcher."` -- the decision is
constrained to an enum, validated against the allowed set in code, and falls
back to a safe default. A model's output never addresses a node directly.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from llm import fast, structured
from tools.file_tools import all_notes

logger = logging.getLogger("router")

Route = Literal["research", "notes"]
VALID_ROUTES: set[str] = {"research", "notes"}
DEFAULT_ROUTE: Route = "research"

ROUTER_SYSTEM = """You route a user's question to one of two handlers.

"notes"    - the question is about material already researched and saved, or
             asks what has been researched so far.
"research" - anything else: a new topic, a request for current information, or
             a follow-up needing evidence the saved notes do not contain.

When the question could plausibly go either way, choose "research". Stale
notes are worse than a redundant search."""


class RouteDecision(BaseModel):
    route: Route = Field(description="Either 'research' or 'notes'")
    reason: str = Field(default="", description="One short clause")


def route(query: str) -> Route:
    """Pick a handler. Never raises, never returns anything unroutable."""
    notes = all_notes()
    saved = ", ".join(n["topic"] for n in notes) if notes else "(nothing saved yet)"

    decision = structured(
        fast(),
        RouteDecision,
        ROUTER_SYSTEM,
        f"Saved research topics: {saved}\n\nQuestion: {query}",
        default=RouteDecision(route=DEFAULT_ROUTE, reason="fallback"),
    )

    chosen = str(decision.route).strip().lower()
    if chosen not in VALID_ROUTES:
        logger.warning("unroutable %r; defaulting to %s", chosen, DEFAULT_ROUTE)
        chosen = DEFAULT_ROUTE

    logger.info("route=%s (%s)", chosen, decision.reason or "no reason given")
    return chosen  # type: ignore[return-value]
