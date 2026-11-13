"""The deep-research graph.

    plan ──► research (xN, parallel) ──► gaps ──┬──► research (follow-ups) ...
                                                └──► report

`plan` decomposes. `research` runs once per sub-question, fanned out with
`Send` so each executes concurrently in its own context. `gaps` is a barrier:
it sees every branch's findings together, decides what is still unsupported,
and either issues narrower follow-up queries or hands off to `report`.

The loop terminates on whichever comes first: no gaps left, the round budget,
the search budget, or the source budget. Explicit budgets rather than a retry
counter, because "how many times have we retried" is not the question you
actually want answered when deciding whether to keep going.
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.gap_analyzer import find_gaps
from agents.planner import plan
from agents.reporter import write_report
from agents.sub_researcher import research_sub_question
from config import settings
from state import ResearchState, SubQuestion

logger = logging.getLogger("graph")


class SubResearchInput(TypedDict):
    """What one fanned-out research branch receives. Not the main state."""
    sub_question: SubQuestion
    brief: str
    round: int
    follow_up_query: str | None


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def plan_node(state: ResearchState) -> dict:
    topic, brief, sub_questions = plan(state["query"])
    return {"topic": topic, "brief": brief, "sub_questions": sub_questions, "round": 1}


def research_node(payload: SubResearchInput) -> dict:
    """One sub-question. Runs concurrently with its siblings.

    Everything it returns goes through a reducer, which is what makes
    concurrent writes safe.
    """
    sub_question, sources, searches = research_sub_question(
        payload["sub_question"],
        payload.get("brief", ""),
        round_index=payload.get("round", 0),
        follow_up_query=payload.get("follow_up_query"),
    )
    return {
        "sub_questions": [sub_question],
        "sources": sources,
        "search_count": searches,
    }


def gaps_node(state: ResearchState) -> dict:
    """Barrier. Every branch has reported; decide whether another round is due."""
    sub_questions = state.get("sub_questions", [])
    gaps = find_gaps(state.get("brief", ""), sub_questions)

    stop_reason = ""
    if not gaps:
        stop_reason = "all sub-questions answered"
    elif state.get("round", 1) >= settings.max_rounds:
        stop_reason = f"round budget reached ({settings.max_rounds})"
    elif state.get("search_count", 0) >= settings.max_searches:
        stop_reason = f"search budget reached ({settings.max_searches})"
    elif len(state.get("sources", [])) >= settings.max_sources:
        stop_reason = f"source budget reached ({settings.max_sources})"

    if stop_reason:
        logger.info("stopping: %s", stop_reason)
        return {"gaps": gaps, "stop_reason": stop_reason}

    logger.info(
        "round %d ended with %d gap(s); starting another",
        state.get("round", 1),
        len(gaps),
    )
    return {"gaps": gaps, "stop_reason": "", "round": state.get("round", 1) + 1}


def report_node(state: ResearchState) -> dict:
    report, issues = write_report(
        state["query"],
        state.get("brief", ""),
        state.get("sub_questions", []),
        state.get("sources", []),
    )
    return {"report": report, "citation_issues": issues, "answer": report}


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

def fan_out(state: ResearchState) -> list[Send]:
    """Send every pending sub-question to its own research branch."""
    brief = state.get("brief", "")
    round_index = state.get("round", 1)
    return [
        Send(
            "research",
            SubResearchInput(
                sub_question=sq,
                brief=brief,
                round=round_index,
                follow_up_query=None,
            ),
        )
        for sq in state.get("sub_questions", [])
    ]


def route_after_gaps(state: ResearchState) -> list[Send] | Literal["report"]:
    """Either issue targeted follow-ups, or write the report.

    Note what gets sent back: the gap's `follow_up_query`, not the original
    sub-question. Re-running the same query is what makes a retry loop
    pointless.
    """
    if state.get("stop_reason"):
        return "report"

    by_id = {sq["id"]: sq for sq in state.get("sub_questions", [])}
    # `round` was already incremented by gaps_node when it chose to continue.
    current_round = state.get("round", 1)

    sends = [
        Send(
            "research",
            SubResearchInput(
                sub_question=by_id[gap["sub_question_id"]],
                brief=state.get("brief", ""),
                round=current_round,
                follow_up_query=gap.get("follow_up_query") or None,
            ),
        )
        for gap in state.get("gaps", [])
        if gap["sub_question_id"] in by_id
    ]
    return sends or "report"


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node, input_schema=SubResearchInput)
    # defer=True makes this a true barrier: it waits for every fanned-out
    # branch, including ones that finish in a later superstep.
    graph.add_node("gaps", gaps_node, defer=True)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["research"])
    graph.add_edge("research", "gaps")
    graph.add_conditional_edges("gaps", route_after_gaps, ["research", "report"])
    graph.add_edge("report", END)

    return graph


def compile_graph(checkpointer=None):
    """Compile with durable checkpointing.

    SqliteSaver rather than MemorySaver: once a run takes minutes rather than
    seconds, losing every checkpoint when the process exits stops being an
    acceptable trade.

    The connection is opened directly rather than through
    `SqliteSaver.from_conn_string`. That helper is a context manager: entering
    it and keeping only the saver leaves nothing referencing the manager, so it
    is finalised immediately and closes the connection out from under the saver
    it just returned. The next checkpoint write then raises "Cannot operate on
    a closed database".
    """
    if checkpointer is None:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(settings.checkpoint_path),
            # Fanned-out branches run on a thread pool; the saver serialises
            # access with its own lock.
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(conn)

    return build_graph().compile(checkpointer=checkpointer)
